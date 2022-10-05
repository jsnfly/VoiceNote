import socket
import wave
import pyaudio
import numpy as np
import time

audio = pyaudio.PyAudio()

# TODO: get from client?
PORT = 12345
CHANNELS = 1
RATE = 44100
FORMAT = pyaudio.paInt16

class Sample:

    def __init__(self, fragments=[]):
        self.fragments = fragments

    def append(self, fragments):
        self.fragments.append(fragments)

    def numpy(self):
        return np.frombuffer(b''.join(self.fragments), dtype=np.int16)

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


def predict():
    time.sleep(0.1)


def main():
    sample = Sample()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((socket.gethostname(), PORT))
        sock.listen()
        conn_sock, conn_addr = sock.accept()
        with conn_sock:
            print('Connected by', conn_addr)
            time.sleep(0.1)
            for i in range(20):
                sample.append(conn_sock.recv(2**17, socket.MSG_DONTWAIT))
                predict()
            sample.to_wav_file('test.wav')



if __name__ == '__main__':
    main()