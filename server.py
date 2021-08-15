import socket
import time
import wave
import pyaudio
import torch
import torchaudio
import numpy as np
from pathlib import Path
from functools import cached_property
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


HOST = '0.0.0.0'
PORT = 65432

WAVE_OUTPUT_FILENAME = "output.wav"
MODEL = 'jonatasgrosman/wav2vec2-large-xlsr-53-german'
RATE = 44_100
CHUNK = 4096
CHANNELS = 1
FORMAT = pyaudio.paInt16

audio = pyaudio.PyAudio()

class Frame:

    def __init__(self, binary_data):
        self._data = binary_data

    @property
    def data(self):
        return self._data

    @cached_property
    def numpy(self):
        return np.frombuffer(self._data, dtype=np.int16)

    @cached_property
    def std(self):
        return self.numpy.std()

class Sample:

    def __init__(self, frames):
        self.frames = frames
        self.frames_per_second = RATE / CHUNK

        # TODO: is not right, because Frames do not have a consitent size (see below).
        self.num_seconds_to_stop = 20

    def to_wav_file(self, file_path):
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join([f.data for f in self.frames]))
        wf.close()

    def to_numpy(self):
        return np.concatenate([f.numpy for f in self.frames])

    def __add__(self, other):
        return self.__class__(self.frames + other.frames)

    def append(self, frame):
        self.frames.append(frame)

    def is_finished(self):
        num_empty_frames_to_stop = int(self.frames_per_second * self.num_seconds_to_stop)
        if len(self.frames) < num_empty_frames_to_stop:
            return False
        return self.frames_are_empty(self.frames[-num_empty_frames_to_stop:])

    def is_empty(self):
        return self.frames_are_empty(self.frames)

    @staticmethod
    def frames_are_empty(frames):
        std_devs = np.array([f.std for f in frames])

        # TODO: Parameter
        return std_devs.max() / std_devs.min() < 15

    

def save_audio_and_prediction(save_path, sample, prediction):
    save_path = Path(save_path) / time.strftime("%Y%m%d-%H%M%S")
    save_path.mkdir(parents=True)
    with open(save_path / 'prediction.txt', 'w') as f:
        f.write(prediction)
    sample.to_wav_file(str(save_path / 'sample.wav'))

processor = Wav2Vec2Processor.from_pretrained(MODEL)
model = Wav2Vec2ForCTC.from_pretrained(MODEL)
resambler = torchaudio.transforms.Resample(RATE, 16_000)

current_sample = Sample([])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind((HOST, PORT))
    sock.listen()
    conn, addr = sock.accept()
    with conn:
        print('Connected by', addr)
        while True:

            # TODO: resulting arrays have inconsistent sizes. The data passed in by the client has correct size.
            # https://stackoverflow.com/questions/1708835/python-socket-receive-incoming-packets-always-have-a-different-size
            current_sample.append(Frame(conn.recv(CHUNK * 16)))
            if current_sample.is_finished():
                if not current_sample.is_empty():
                    data = current_sample.to_numpy()
                    if RATE != 16_000:
                        data = resambler(torch.Tensor(data)).numpy().squeeze()
                    inputs = processor(data, sampling_rate=16_000, return_tensors="pt", padding=True)
                    with torch.no_grad():
                        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    prediction = processor.batch_decode(predicted_ids)[0]
                    print("Prediction:", prediction)
                    save_audio_and_prediction('outputs', current_sample, prediction)
                    conn.sendall(bytes(prediction, 'utf-8'))
                current_sample = Sample([])
