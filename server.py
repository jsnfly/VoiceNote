import asyncio
from functools import cached_property
import wave
import torch
import pyaudio
import numpy as np
import time
import whisper
from torchaudio.transforms import Resample

PORT = 12345
MAXIMUM_PREDICTION_FREQ = 1.  # Predictions/Second
SAMPLE_OVERLAP = 0.5  # Final seconds of current sample to be used in the next sample to prevent losing speech segments


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
        result = whisper.decode(model, self.mel_spectrogram, options)
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
        mel = whisper.log_mel_spectrogram(padded).to(model.device)
        return mel

    @cached_property
    def resampler(self):
        return Resample(self.rate, 16_000)

    @property
    def is_finished(self):
        return self._is_finished

    @property
    def is_empty(self):
        return self.result is not None and self.result.no_speech_prob > 0.7

    def to_wav_file(self, file_path, channels, format):
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.fragments))
        wf.close()


def predict(sample):
    sample.transcribe(model, options)
    if not sample.is_empty:
        print(sample.result.text)


async def read(reader):
    return await reader.read(2**42)  # Large number to read the whole buffer.

import json

async def handle_connection(reader, writer):
    audio_config = json.loads(await read(reader))
    writer.write(b'OK')
    assert audio_config['format'] == pyaudio.paInt16
    bytes_per_second = audio_config['rate'] * 2  # Times 2 because each data point has 16 bits.

    sample = Sample([], audio_config['rate'])
    while not reader.at_eof():
        start = time.time()
        sample.append(await read(reader))
        predict(sample)
        if sample.is_finished or sample.is_empty:
            if not sample.is_empty:
                print('FINISHED')
                print(sample.result.text)

            initial_fragment = b''.join(sample.fragments)[-int(SAMPLE_OVERLAP * bytes_per_second):]
            sample = Sample([initial_fragment], audio_config['rate'])
        end = time.time()
        if (diff := 1 / MAXIMUM_PREDICTION_FREQ - (end - start)) > 0:
            await asyncio.sleep(diff)
    print('Connection closed.')

async def run_server():
    server = await asyncio.start_server(handle_connection, '0.0.0.0', PORT)
    async with server:
        await server.serve_forever()


async def main():
    await run_server()


if __name__ == '__main__':
    audio = pyaudio.PyAudio()
    model = whisper.load_model('base', device='cuda')
    options = whisper.DecodingOptions(language='de')

    # asyncio is currently not really needed but it makes some things a bit cleaner (e.g. setting up the connection/
    # receiving the data)
    asyncio.run(main())
