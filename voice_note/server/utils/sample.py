import wave
import torch
import time
from pathlib import Path
from torchaudio.transforms import Resample
from typing import List, Union

from server.utils.audio import AudioConfig
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class Sample:

    def __init__(self, fragments: List[bytes], audio_config: AudioConfig):
        self.fragments = fragments
        self.audio_config = audio_config
        self.resampler = Resample(audio_config.rate, 16_000)
        self.result = None

    def transcribe(self, model: WhisperForConditionalGeneration, processor: WhisperProcessor):
        if len(self.fragments) == 0:
            return

        input_features = processor(self.audio_data, sampling_rate=16_000, return_tensors="pt").input_features
        pred_ids = model.generate(input_features.to(model.device, dtype=model.dtype))
        self.result = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

    @property
    def audio_data(self) -> torch.Tensor:
        data = torch.asarray(b''.join(self.fragments), dtype=torch.int16).float()

        # Is also done in OpenAI's whisper implementation in whisper#load_audio and seems to make data similar to the
        # result of that.
        data /= 32768.

        return self.resampler(data)

    def save(self, save_dir: Union[Path, str]) -> Path:
        assert self.result is not None, "Please call `.transcribe` first"

        save_path = Path(save_dir) / time.strftime("%Y%m%d-%H%M%S")
        save_path.mkdir(parents=True)
        with open(save_path / 'prediction.txt', 'w') as f:
            f.write(self.result)
        self.to_wav_file(str(save_path / 'sample.wav'))
        return save_path

    def to_wav_file(self, file_path: str):
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(self.audio_config.channels)
        wf.setsampwidth(self.audio_config.sample_size)
        wf.setframerate(self.audio_config.rate)

        wf.writeframes(b''.join(self.fragments))
        wf.close()
