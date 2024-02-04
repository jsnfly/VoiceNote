import wave
import torch
import whisper
import time
from pathlib import Path
from torchaudio.transforms import Resample
from whisper.decoding import DecodingTask


class Sample:

    def __init__(self, fragments, audio_config):
        self.fragments = fragments
        self.audio_config = audio_config
        self.resampler = Resample(audio_config.rate, 16_000)
        self.result: whisper.DecodingResult = None

    def append(self, fragment):
        self.fragments.append(fragment)

    def transcribe(self, model, options):
        if len(self.fragments) == 0:
            return

        if not hasattr(self, 'decoding_task'):
            self.decoding_task = DecodingTask(model, options)

        self.result = self.decoding_task.run(self.mel_spectrogram.unsqueeze(0).to(model.device))[0]

    @property
    def mel_spectrogram(self):
        data = torch.frombuffer(b''.join(self.fragments), dtype=torch.int16).float()
        data /= 32768.  # Is also done in whisper#load_audio and seems to make data similar to the result of that.
        resampled = self.resampler(data)
        padded = whisper.pad_or_trim(resampled)
        return whisper.log_mel_spectrogram(padded)

    def save(self, save_dir):
        assert self.result is not None, "Please call `.transcribe` first"

        save_path = Path(save_dir) / time.strftime("%Y%m%d-%H%M%S")
        save_path.mkdir(parents=True)
        with open(save_path / 'prediction.txt', 'w') as f:
            f.write(self.result.text)
        self.to_wav_file(str(save_path / 'sample.wav'))
        return save_path

    def to_wav_file(self, file_path):
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(self.audio_config.channels)
        wf.setsampwidth(self.audio_config.sample_size)
        wf.setframerate(self.audio_config.rate)

        data = b''.join(self.fragments)
        wf.writeframes(data)
        wf.close()
