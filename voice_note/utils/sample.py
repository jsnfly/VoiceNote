import wave
import torch
import whisper
import time
from pathlib import Path
from torchaudio.transforms import Resample
from whisper.decoding import DecodingTask

class Sample:

    def __init__(self, fragments, rate):
        self.fragments = fragments

        self.resampler = Resample(rate, 16_000)
        self.decoding_task = None

        self.result: whisper.DecodingResult = None
        self.time_of_last_transcription_change = None
        self._is_finished = False

    def append(self, fragment):
        self.fragments.append(fragment)

    def transcribe(self, model, options):
        if self.decoding_task is None:
            self.decoding_task = DecodingTask(model, options)
        result = self.decoding_task.run(self.mel_spectrogram.unsqueeze(0).to(model.device))[0]
        last_token = result.tokens[-1]
        if self.result is None or self.result.tokens[-1] != last_token:
            # If the final timestamp is different there was additional speech added since last transcription.
            # Sometimes the final token is not a timestamp. This is also a sign that the transcription is still
            # changing.
            self.time_of_last_transcription_change = time.time()
        elif time.time() - self.time_of_last_transcription_change > 2:
            # If no speech was added for two seconds the sample is assumed to be finished.
            self._is_finished = True
        self.result = result

    @property
    def mel_spectrogram(self):
        data = torch.frombuffer(b''.join(self.fragments), dtype=torch.int16).float()
        data /= 32768.  # Is also done in whisper#load_audio and seems to make data similar to the result of that.
        resampled = self.resampler(data)
        padded = whisper.pad_or_trim(resampled)
        return whisper.log_mel_spectrogram(padded)

    @property
    def is_finished(self):
        return self._is_finished

    @property
    def is_empty(self):
        return self.result is not None and self.result.no_speech_prob > 0.7

    def save(self, save_dir, channels, sample_size):
        assert self.result is not None, "Please call `.transcribe` first"

        save_path = Path(save_dir) / time.strftime("%Y%m%d-%H%M%S")
        save_path.mkdir(parents=True)
        with open(save_path / 'prediction.txt', 'w') as f:
            f.write(self.result.text)
        self.to_wav_file(str(save_path / 'sample.wav'), channels, sample_size)


    def to_wav_file(self, file_path, channels, sample_size):
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(sample_size)
        wf.setframerate(self.resampler.orig_freq)

        data = b''.join(self.fragments)

        # 1 second is added to account for inaccuracies.
        speech_duration = (self.last_token - self.decoding_task.tokenizer.timestamp_begin) * 0.02 + 1

        num_speech_bytes = min(len(data), int(speech_duration * self.resampler.orig_freq * sample_size))
        wf.writeframes(data[:num_speech_bytes])
        wf.close()


    @property
    def last_token(self):
        return (None if self.result is None else self.result.tokens[-1])
