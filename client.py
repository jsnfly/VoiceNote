import socket
import time
import pyaudio

FORMAT = pyaudio.paInt16  # https://en.wikipedia.org/wiki/Audio_bit_depth

# TODO: get from device_info?
CHANNELS = 1
RATE = 44_100

HOST = socket.gethostname()


def connect(sock):
    while True:
        try:
            sock.connect((HOST, 12345))
            print('Connected.')
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
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
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