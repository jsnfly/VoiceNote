from types import SimpleNamespace

import pyaudio
import pytest
import torch
from transformers import StoppingCriteriaList

from server.utils.audio import AudioConfig
from server.utils.sample import Sample, _stitch

SAMPLE_RATE = 16_000


class FakeProcessor:
    """Encodes the raw window length into the features so the fake model can report it back."""

    def __call__(self, audio, sampling_rate, return_tensors):
        assert sampling_rate == SAMPLE_RATE
        return SimpleNamespace(input_features=torch.tensor([[float(audio.shape[-1])]]))

    def batch_decode(self, pred_ids, skip_special_tokens=True):
        return [str(int(pred_ids.item()))]


class FakeModel:
    device, dtype = 'cpu', torch.float32

    def __init__(self):
        self.window_lengths = []

    def generate(self, input_features, stopping_criteria=None):
        length = int(input_features.item())
        self.window_lengths.append(length)
        return torch.tensor([[float(length)]])


class CancelledCriteria:
    def __call__(self, input_ids, scores, **kwargs):
        return True


def make_sample(seconds: float) -> Sample:
    config = AudioConfig(pyaudio.paInt16, 1, SAMPLE_RATE)
    fragments = [b'\x00\x00' * int(SAMPLE_RATE * seconds)]
    return Sample(fragments, config)


def test_stitch_drops_word_overlap():
    assert _stitch('', 'hello world') == 'hello world'
    assert _stitch('hello world', 'world peace') == 'hello world peace'
    assert _stitch('a b c', 'd e f') == 'a b c d e f'


def test_transcribe_single_window_for_short_audio():
    model, sample = FakeModel(), make_sample(10)
    sample.transcribe(model, FakeProcessor())
    assert model.window_lengths == [SAMPLE_RATE * 10]


def test_transcribe_splits_long_audio_into_overlapping_windows():
    model, sample = FakeModel(), make_sample(75)
    window = 30 * SAMPLE_RATE
    overlap = 1 * SAMPLE_RATE

    sample.transcribe(model, FakeProcessor())

    # 75s audio: [0:30s], [29s:59s], [58s:75s] windows (1s overlap between neighbors).
    assert model.window_lengths == [window, window, 75 * SAMPLE_RATE - (2 * (window - overlap))]


def test_transcribe_cancellation_stops_before_first_window():
    model, sample = FakeModel(), make_sample(10)
    sample.transcribe(model, FakeProcessor(), StoppingCriteriaList([CancelledCriteria()]))
    assert model.window_lengths == []
