import wave
import torch
import whisper
import time
from pathlib import Path
from torchaudio.transforms import Resample
from whisper.decoding import DecodingTask

torch.set_num_threads(1)


class Sample:

    def __init__(self, fragments, rate, channels, sample_size):
        self.fragments = fragments
        self.rate = rate
        self.channels = channels
        self.sample_size = sample_size

        self.resampler = Resample(rate, 16_000)

        self.result: whisper.DecodingResult = None
        self.time_of_last_transcription_change = None
        self._finished = False

    def append(self, fragment):
        self.fragments.append(fragment)

    def transcribe(self, model, options):
        if not len(self):
            return
        if not hasattr(self, 'decoding_task'):
            self.decoding_task = DecodingTask(model, options)

        new_result = self.decoding_task.run(self.mel_spectrogram.unsqueeze(0).to(model.device))[0]

        if self.last_token != new_result.tokens[-1]:
            # If the final timestamp is different there was additional speech added since last transcription.
            # Sometimes the final token is not a timestamp. This is also a sign that the transcription is still
            # changing.
            self.time_of_last_transcription_change = time.time()
        elif time.time() - self.time_of_last_transcription_change > 2:
            # If no speech was added for two seconds the sample is assumed to be finished.
            self._finished = True
        self.result = new_result

    @property
    def last_token(self):
        return (None if self.result is None else self.result.tokens[-1])

    def __len__(self):
        return len(b''.join(self.fragments))

    @property
    def mel_spectrogram(self):
        data = torch.frombuffer(b''.join(self.fragments), dtype=torch.int16).float()
        data /= 32768.  # Is also done in whisper#load_audio and seems to make data similar to the result of that.
        resampled = self.resampler(data)
        padded = whisper.pad_or_trim(resampled)
        return whisper.log_mel_spectrogram(padded)

    @property
    def finished(self):
        return self._finished

    @property
    def is_empty(self):
        return self.result is not None and self.result.no_speech_prob > 0.7

    def save(self, save_dir):
        assert self.result is not None, "Please call `.transcribe` first"

        save_path = Path(save_dir) / time.strftime("%Y%m%d-%H%M%S")
        save_path.mkdir(parents=True)
        with open(save_path / 'prediction.txt', 'w') as f:
            f.write(self.result.text)
        self.to_wav_file(str(save_path / 'sample.wav'))

    def to_wav_file(self, file_path):
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.sample_size)
        wf.setframerate(self.resampler.orig_freq)

        data = b''.join(self.fragments)

        # 1 second is added to account for inaccuracies.
        speech_duration = (self.last_token - self.decoding_task.tokenizer.timestamp_begin) * 0.02 + 1

        num_speech_bytes = min(len(data), int(speech_duration * self.resampler.orig_freq * self.sample_size))
        wf.writeframes(data[:num_speech_bytes])
        wf.close()
