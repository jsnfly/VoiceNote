import wave
import torch
import whisper
import time
from functools import cached_property
from torchaudio.transforms import Resample

class Sample:

    def __init__(self, fragments, rate):
        self.fragments = fragments
        self.rate = rate
        self.result: whisper.DecodingResult = None
        self.time_of_last_transcription_change = None
        self._is_finished = False

    def append(self, fragments):
        self.fragments.append(fragments)

    def transcribe(self, model, options):
        result = whisper.decode(model, self.mel_spectrogram.to(model.device), options)
        if self.result is None or self.result.text != result.text:
            self.time_of_last_transcription_change = time.time()
        elif time.time() - self.time_of_last_transcription_change > 2:
            self._is_finished = True
        self.result = result

    @property
    def mel_spectrogram(self):
        data = torch.frombuffer(b''.join(self.fragments), dtype=torch.int16).float()
        data /= 32768.  # Is also done in whisper#load_audio and seems to make data similar to the result of that.
        resampled = self.resampler(data)
        padded = whisper.pad_or_trim(resampled)
        return whisper.log_mel_spectrogram(padded)

    @cached_property
    def resampler(self):
        return Resample(self.rate, 16_000)

    @property
    def is_finished(self):
        return self._is_finished

    @property
    def is_empty(self):
        return self.result is not None and self.result.no_speech_prob > 0.7

    def to_wav_file(self, file_path, channels, sample_size):
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(sample_size)
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.fragments))
        wf.close()
