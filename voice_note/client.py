import socket
import time
import pyaudio
import json
import argparse
from functools import lru_cache


@lru_cache(maxsize=1)
def get_audio_config(input_device_index):
    if input_device_index is None:
        device_config = audio.get_default_input_device_info()
    else:
        device_config = audio.get_device_info_by_index(input_device_index)

    return {
        'format': pyaudio.paInt16,  # https://en.wikipedia.org/wiki/Audio_bit_depth,
        'channels': 1,
        'rate': int(device_config['defaultSampleRate'])
    }


def connect(sock, host, port, input_device_index):
    while True:
        try:
            sock.connect((host, port))
            print('Connected.')
            sock.sendall(json.dumps(get_audio_config(input_device_index)).encode())
            data = sock.recv(64)
            assert data == b'OK'
            print('Initialized.')
            break
        except ConnectionRefusedError:
            print('Trying to connect...')
            time.sleep(1.0)


def main(host, port, input_device_index):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        connect(sock, host, port, input_device_index)

        def _callback(in_data, frame_count, time_info, status):
            message = pyaudio.paContinue
            try:
                sock.sendall(in_data)
            except BrokenPipeError:
                message = pyaudio.paComplete
            return (in_data, message)

        stream = audio.open(
            **get_audio_config(input_device_index),
            input=True,
            frames_per_buffer=0,
            input_device_index=input_device_index,
            stream_callback=_callback
        )

        print("* recording")
        while stream.is_active():
            time.sleep(0.1)

        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default='0.0.0.0')
    parser.add_argument("--port", default=12345, type=int)
    parser.add_argument("--input-device-index", type=int)

    args = parser.parse_args()
    audio = pyaudio.PyAudio()
    main(args.host, args.port, args.input_device_index)
