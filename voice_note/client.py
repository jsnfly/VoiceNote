import socket
import time
import pyaudio
import json
import argparse
from functools import lru_cache
from utils import recv_messages, send_message


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
            sock.setblocking(0)
            print('Connected.')
            send_message(get_audio_config(input_device_index), sock)
            messages, _ = recv_messages(sock)
            assert messages[0]['response'] == 'OK'
            print('Initialized.')
            break
        except ConnectionRefusedError:
            print('Trying to connect...')
            time.sleep(1.0)


def receive(sock):
    try:
        result = b''
        while True:
            try:
                result += sock.recv(2**20)
                result = json.loads(result.decode())
                break
            except json.decoder.JSONDecodeError:
                pass
        return result
    except BlockingIOError:
        pass


def main(host, port, input_device_index):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        connect(sock, host, port, input_device_index)

        def _callback(in_data, frame_count, time_info, status):
            message = pyaudio.paContinue
            try:
                sock.sendall(in_data)
                result = receive(sock)
                if result is not None:
                    print(result['text'])
                    if 'audio' in result:
                        audio_res = result['audio']
                        out_stream = audio.open(format=audio.get_format_from_width(audio_res['width']),
                                                channels=audio_res['channels'], rate=audio_res['rate'], output=True)
                        # TODO: get rid of eval
                        # TODO: gets picked up by microphone.
                        out_stream.write(eval(audio_res['frames']))
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
