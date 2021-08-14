import argparse
import pyaudio
import socket
import time

parser = argparse.ArgumentParser(description='Speech to text client.')
parser.add_argument('--host')
parser.add_argument('--port', type=int)
parser.add_argument('--input_device_index', type=int, default=None)
args = parser.parse_args()

CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44_100

audio = pyaudio.PyAudio()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    def callback(in_data, frame_count, time_info, status):
        message = pyaudio.paContinue
        try:
            sock.sendall(in_data)
            # prediction = sock.recv(4096)
            # if prediction:
            #     print("Prediction:", prediction)
            #     message = pyaudio.paComplete
        except BrokenPipeError:
            message = pyaudio.paComplete
        return (in_data, message)
    while True:
        try:
            sock.connect((args.host, args.port))
            print('Connected')
            break
        except ConnectionRefusedError:
            print('Trying to connect.')
            time.sleep(1.0)
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=args.input_device_index,
        stream_callback=callback
    )
    print("* recording")

    while stream.is_active():
        time.sleep(0.01)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    