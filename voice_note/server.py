import asyncio
import pyaudio
import time
import json
import whisper
from sample import Sample

PORT = 12345
MAXIMUM_PREDICTION_FREQ = 1.  # Predictions/Second
SAMPLE_OVERLAP = 0.5  # Final seconds of current sample to be used in the next sample to prevent losing speech segments
SAVE_PREDICTIONS = True


def predict(sample):
    sample.transcribe(model, options)
    if not sample.is_empty:
        print(sample.result.text, end="\r")


async def initialize(reader, writer):
    audio_config = json.loads(await read(reader))
    writer.write(b'OK')
    return audio_config


async def read(reader):
    return await reader.read(2**42)  # Large number to read the whole buffer.


def finish_sample(sample, audio_config):
    if not sample.is_empty:
        if SAVE_PREDICTIONS:
            sample.save('outputs', audio_config['channels'], audio.get_sample_size(audio_config['format']))
        print("\nFinished: ", sample.result.text)
    bytes_per_second = audio_config['rate'] * 2  # Times 2 because each data point has 16 bits.
    initial_fragment = b''.join(sample.fragments)[-int(SAMPLE_OVERLAP * bytes_per_second):]
    return Sample([initial_fragment], audio_config['rate'])


async def handle_connection(reader, writer):
    audio_config = await initialize(reader, writer)
    assert audio_config['format'] == pyaudio.paInt16

    sample = Sample([], audio_config['rate'])
    while not reader.at_eof():
        start = time.time()
        sample.append(await read(reader))
        predict(sample)
        if sample.is_finished or sample.is_empty:
            sample = finish_sample(sample, audio_config)
        end = time.time()
        if (diff := 1 / MAXIMUM_PREDICTION_FREQ - (end - start)) > 0:
            await asyncio.sleep(diff)
    print('Connection closed.')


async def main():
    server = await asyncio.start_server(handle_connection, '0.0.0.0', PORT)
    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    audio = pyaudio.PyAudio()
    model = whisper.load_model('base', device='cuda')
    options = whisper.DecodingOptions(language='de')

    # asyncio is currently not really needed but it makes some things a bit cleaner (e.g. setting up the connection/
    # receiving the data)
    asyncio.run(main())
