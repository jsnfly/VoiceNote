import torch
from torchaudio.transforms import Resample
from typing import List
from transformers import StoppingCriteriaList, WhisperForConditionalGeneration, WhisperProcessor

from server.utils.audio import AudioConfig

SAMPLE_RATE = 16_000
# Whisper's decoder consumes a fixed 30s input window, so longer recordings are split into
# overlapping windows that are transcribed independently and stitched back together.
WINDOW_SECONDS = 30
OVERLAP_SECONDS = 1


def _stitch(previous: str, nxt: str) -> str:
    """Merges two window transcriptions, dropping words duplicated by the window overlap."""
    if not previous:
        return nxt
    prev_words = previous.split()
    next_words = nxt.split()
    for n in range(min(len(prev_words), len(next_words), 8), 0, -1):
        if prev_words[-n:] == next_words[:n]:
            return previous + ' ' + ' '.join(next_words[n:])
    return previous + ' ' + nxt


class Sample:

    def __init__(self, fragments: List[bytes], audio_config: AudioConfig):
        self.fragments = fragments
        self.audio_config = audio_config
        self.resampler = Resample(audio_config.rate, 16_000)
        self.result = None

    @property
    def audio_data(self) -> torch.Tensor:
        data = torch.asarray(self.get_audio_bytes(), dtype=torch.int16).float()

        # Is also done in OpenAI's whisper implementation in whisper#load_audio and seems to make data similar to the
        # result of that.
        data /= 32768.

        return self.resampler(data)

    def get_audio_bytes(self) -> bytes:
        return b''.join(self.fragments)

    def transcribe(self, model: WhisperForConditionalGeneration, processor: WhisperProcessor,
                   stopping_criteria: StoppingCriteriaList = None):
        if len(self.fragments) == 0:
            return

        audio = self.audio_data
        window = WINDOW_SECONDS * SAMPLE_RATE
        overlap = OVERLAP_SECONDS * SAMPLE_RATE

        result = ''
        for start in range(0, len(audio), window - overlap):
            if stopping_criteria is not None and stopping_criteria(torch.zeros(1, dtype=torch.long), None):
                break

            input_features = processor(audio[start:start + window], sampling_rate=SAMPLE_RATE,
                                       return_tensors='pt').input_features
            pred_ids = model.generate(input_features.to(model.device, dtype=model.dtype),
                                      stopping_criteria=stopping_criteria)
            result = _stitch(result, processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip())

        self.result = result
