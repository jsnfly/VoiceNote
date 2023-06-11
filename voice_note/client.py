import pyaudio
import socket
import time
import argparse
from functools import lru_cache, partial
from utils import audio, recv_messages, send_message, send_data
from utils.misc import prepare_log_file, log_bytes
from client_config import AUDIO_FORMAT, NUM_CHANNELS

BYTES_LOG_FILE = 'logs/client_bytes.log'
prepare_log_file(BYTES_LOG_FILE)


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


@lru_cache(maxsize=1)
def get_audio_config(input_device_index):
    if input_device_index is None:
        device_config = audio.get_default_input_device_info()
    else:
        device_config = audio.get_device_info_by_index(input_device_index)

    return {
        'format': AUDIO_FORMAT,
        'channels': NUM_CHANNELS,
        'rate': int(device_config['defaultSampleRate'])
    }


def get_stream(sock, input_device_index):
    stream = audio.open(
            **get_audio_config(input_device_index),
            input=True,
            frames_per_buffer=pyaudio.paFramesPerBufferUnspecified,
            input_device_index=input_device_index,
            stream_callback=get_callback(sock)
        )
    return stream


def get_callback(sock):
    return partial(_callback, sock)


def _callback(sock, in_data, *args):
    try:
        send_data(in_data, sock)
        if BYTES_LOG_FILE is not None:
            with open(BYTES_LOG_FILE, 'a') as f:
                log_bytes(in_data, f)
        messages, _ = recv_messages(sock, blocking=False)
        for msg in messages:
            if 'text' in msg:
                print(msg['text'])
            if 'audio' in msg:
                msg_audio = msg['audio']
                out_stream = audio.open(format=audio.get_format_from_width(msg_audio['width']),
                                        channels=msg_audio['channels'], rate=msg_audio['rate'], output=True)
                # TODO: gets picked up by microphone.
                out_stream.write(msg_audio['frames'])
                out_stream.close()
        return _, pyaudio.paContinue
    except BrokenPipeError:
        return _, pyaudio.paComplete


def main(host, port, input_device_index):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        connect(sock, host, port, input_device_index)
        stream = get_stream(sock, input_device_index)

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
    main(args.host, args.port, args.input_device_index)
