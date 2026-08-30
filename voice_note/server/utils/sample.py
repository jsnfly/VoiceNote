import torch
from torchaudio.transforms import Resample
from threading import Thread
from typing import List

from server.utils.audio import AudioConfig
from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer


class Sample:

    def __init__(self, fragments: List[bytes], audio_config: AudioConfig, language: str = 'auto'):
        self.fragments = fragments
        self.audio_config = audio_config
        self.language = language
        self.resampler = Resample(audio_config.rate, 16_000)
        self.result = None
        self.streaming_fragments: List[str] = []

    @property
    def audio_data(self) -> torch.Tensor:
        data = torch.asarray(self.get_audio_bytes(), dtype=torch.int16).float()

        # Is also done in OpenAI's whisper implementation in whisper#load_audio and seems to make data similar to the
        # result of that.
        data /= 32768.

        return self.resampler(data)

    def get_audio_bytes(self) -> bytes:
        return b''.join(self.fragments)

    def transcribe(self, model: AutoModelForRNNT, processor: AutoProcessor, language: str = None):
        if len(self.fragments) == 0:
            return ''

        lang = language or self.language
        self._transcribe_streaming(model, processor, lang)
        self.result = self._transcribe_offline(model, processor, lang)
        return self.result

    def _transcribe_streaming(self, model: AutoModelForRNNT, processor: AutoProcessor, language: str) -> None:
        """ Cache-aware streaming pass over chunked mel features. Yields the interim text fragments internally;
        the offline pass below produces the authoritative transcript (streaming chunks omit trailing punctuation).
        """
        fe = processor.feature_extractor
        sr = fe.sampling_rate
        hop_length, n_fft = fe.hop_length, fe.n_fft

        audio = self.audio_data
        n_samples_first = processor.num_samples_first_audio_chunk
        if len(audio) <= n_samples_first:
            # Too short for even a single streaming chunk; everything is handled by the offline pass.
            return

        num_lookahead_tokens = processor.default_num_lookahead_tokens

        def make_chunks():
            # The extractor applies preemphasis to each input individually and omits it on the first sample.
            # Preemphasis is applied globally here instead; disabling it keeps windows frame-exact w.r.t. an
            # offline pass over the full utterance.
            preemphasis_backup = fe.preemphasis
            fe.preemphasis = None

            try:
                first_in = processor(audio[:n_samples_first], sampling_rate=sr, is_streaming=True,
                                     is_first_audio_chunk=True, language=language, return_tensors='pt')
                yield first_in.input_features[:, :processor.num_mel_frames_first_audio_chunk, :]

                mel_frame_idx = processor.num_mel_frames_first_audio_chunk
                start_idx = mel_frame_idx * hop_length - n_fft // 2
                nsq = processor.num_samples_per_audio_chunk

                while True:
                    window = audio[start_idx:start_idx + nsq]
                    is_last = start_idx + nsq >= len(audio)

                    if is_last:
                        padded = torch.zeros(nsq)
                        padded[:len(window)] = window
                        window = padded

                    inp = processor(window, sampling_rate=sr, is_streaming=True,
                                    is_first_audio_chunk=False, language=language, return_tensors='pt')
                    yield inp.input_features

                    if is_last:
                        break
                    mel_frame_idx += processor.num_mel_frames_per_audio_chunk
                    start_idx = mel_frame_idx * hop_length - n_fft // 2
            finally:
                fe.preemphasis = preemphasis_backup

        streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
        generation_error = []

        def generate():
            try:
                model.generate(input_features=make_chunks(),
                               num_lookahead_tokens=num_lookahead_tokens,
                               streamer=streamer)
            except Exception as e:
                generation_error.append(e)
                streamer.end()  # Unblock the reader thread.

        thread = Thread(target=generate, daemon=True)
        thread.start()
        try:
            for fragment in streamer:
                if fragment:
                    self.streaming_fragments.append(fragment)
        finally:
            thread.join()
        if generation_error:
            raise generation_error[0]

    @torch.inference_mode()
    def _transcribe_offline(self, model: AutoModelForRNNT, processor: AutoProcessor, language: str) -> str:
        audio = self.audio_data.numpy()
        inputs = processor(audio, sampling_rate=processor.feature_extractor.sampling_rate,
                           language=language, return_tensors='pt')
        output = model.generate(**inputs.to(model.device, dtype=model.dtype), return_dict_in_generate=True)
        decoded = processor.decode(output.sequences, skip_special_tokens=True)
        return decoded[0].strip() if isinstance(decoded, list) else decoded.strip()
