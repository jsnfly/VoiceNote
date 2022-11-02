import asyncio
import wave
import torch
import pyaudio
import numpy as np
import time
import whisper
from torchaudio.transforms import Resample


audio = pyaudio.PyAudio()

# TODO: get from client?
PORT = 12345
CHANNELS = 1
RATE = 44100
FORMAT = pyaudio.paInt16

model = whisper.load_model('base', device='cuda')
options = whisper.DecodingOptions(language='de')
resampler = Resample(RATE, 16_000)


class Sample:

    def __init__(self, fragments=[]):
        self.fragments = fragments
        self.result: whisper.DecodingResult = None
        self._is_finished = False

    def append(self, fragments):
        self.fragments.append(fragments)

    def numpy(self):
        return np.frombuffer(b''.join(self.fragments), dtype=np.int16)

    def torch(self):
        return torch.frombuffer(b''.join(self.fragments), dtype=torch.int16)

    def transcribe(self, model, options):
        result = whisper.decode(model, self.preprocessed, options)
        print()
        if self.result is not None and self.result.text == result.text:
            self._is_finished = True
        self.result = result

    @property
    def preprocessed(self):
        data = self.torch().float() / 32768.  # Is also done in whisper#load_audio and seems to make data similar.
        resampled = resampler(data)
        padded = whisper.pad_or_trim(resampled)
        mel = whisper.log_mel_spectrogram(padded).to(model.device)
        return mel

    @property
    def is_finished(self):
        return self._is_finished

    @property
    def is_empty(self):
        return self.result is not None and self.result.no_speech_prob > 0.7

    def to_wav_file(self, file_path):
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.fragments))
        wf.close()


def predict(sample):
    sample.transcribe(model, options)


async def read(reader):
    return await reader.read(2**42)  # Large number to read the whole buffer.


async def handle_connection(reader, writer):
    sample = Sample()
    while not reader.at_eof():
        start = time.time()
        sample.append(await read(reader))
        predict(sample)
        if sample.is_finished or sample.is_empty:
            if not sample.is_empty:
                print(sample.result.text)
            sample = Sample(fragments=[b''.join(sample.fragments)[-RATE:]])  # Half a second (TODO: make parameter)
        end = time.time()
        if (diff := 1.5 - (end - start)) > 0:
            # Only perform at most one prediction every X seconds. (TODO: make parameter)
            await asyncio.sleep(diff)
    print('Connection closed.')

async def run_server():
    server = await asyncio.start_server(handle_connection, '0.0.0.0', PORT)
    async with server:
        await server.serve_forever()


async def main():
    await run_server()


if __name__ == '__main__':

    # asyncio is currently not really needed but it makes some things a bit cleaner (e.g. setting up the connection/
    # receiving the data)
    asyncio.run(main())
