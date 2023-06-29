import pyaudio
from dataclasses import dataclass, field

audio = pyaudio.PyAudio()


@dataclass
class AudioConfig:
    format: int
    channels: int
    rate: int
    sample_size: int = field(init=False)
    bytes_per_second: int = field(init=False)

    def __post_init__(self):
        self.sample_size = audio.get_sample_size(self.format)
        self.bytes_per_second = self.rate * self.sample_size
