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

model = whisper.load_model("base", device='cuda')
options = whisper.DecodingOptions()
resampler = Resample(RATE, 16_000)

class Sample:

    def __init__(self, fragments=[]):
        self.fragments = fragments

    def append(self, fragments):
        self.fragments.append(fragments)

    def numpy(self):
        return np.frombuffer(b''.join(self.fragments), dtype=np.int16)

    def torch(self):
        return torch.frombuffer(b''.join(self.fragments), dtype=torch.int16)

    def is_finished(self):
        pass

    def is_empty(self):
        pass

    def to_wav_file(self, file_path):
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.fragments))
        wf.close()


def predict(sample):
    print('Predicting...')
    return whisper.decode(model, preprocess(sample), options)


def preprocess(sample):
    data = sample.torch().float() / 32768.  # Is also done in whisper#load_audio and seems to make data similar.
    resampled = resampler(data)
    padded = whisper.pad_or_trim(resampled)
    mel = whisper.log_mel_spectrogram(padded).to(model.device)
    return mel


async def read(reader):
    print('Reading...')
    return await reader.read(2**42)  # Large number to read the whole buffer.


async def handle_connection(reader, writer):
    sample = Sample()
    while not reader.at_eof():
        start = time.time()
        sample.append(await read(reader))
        result = predict(sample)
        if result.no_speech_prob > 0.5:
            sample = Sample()
        else:
            print(result.text)
        end = time.time()
        if (diff := 1 - (end - start)) > 0:
            # Only perform at most one prediction per second.
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
