import pyaudio
import socket
import time
import argparse
from functools import lru_cache
from utils import audio, recv_messages, send_message, send_data
from client_config import AUDIO_FORMAT, NUM_CHANNELS


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


def main(host, port, input_device_index):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        connect(sock, host, port, input_device_index)

        def _callback(in_data, frame_count, time_info, status):
            try:
                send_data(in_data, sock)
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

        stream = audio.open(
            **get_audio_config(input_device_index),
            input=True,
            frames_per_buffer=pyaudio.paFramesPerBufferUnspecified,
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
    main(args.host, args.port, args.input_device_index)
