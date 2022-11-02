import socket
import time
import pyaudio
import json

HOST = '0.0.0.0'
PORT = 12345

audio_config = {
    'format': pyaudio.paInt16,  # https://en.wikipedia.org/wiki/Audio_bit_depth
    'channels': 1,
    'rate': 44_100
}


def connect(sock):
    while True:
        try:
            sock.connect((HOST, PORT))
            print('Connected.')
            sock.sendall(json.dumps(audio_config).encode())
            data = sock.recv(64)
            assert data == b'OK'
            print('Initialized.')
            break
        except ConnectionRefusedError:
            print('Trying to connect...')
            time.sleep(1.0)


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        connect(sock)

        def _callback(in_data, frame_count, time_info, status):
            message = pyaudio.paContinue
            try:
                sock.sendall(in_data)
            except BrokenPipeError:
                message = pyaudio.paComplete
            return (in_data, message)

        stream = audio.open(
            **audio_config,
            input=True,
            frames_per_buffer=0,
            input_device_index=None,
            stream_callback=_callback
        )

        print("* recording")
        while stream.is_active():
            time.sleep(0.1)

        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == '__main__':
    audio = pyaudio.PyAudio()
    main()
