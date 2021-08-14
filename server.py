import socket
import time
import wave
import pyaudio
import torch
import torchaudio
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


HOST = '0.0.0.0'
PORT = 65432

WAVE_OUTPUT_FILENAME = "output.wav"
MODEL = 'jonatasgrosman/wav2vec2-large-xlsr-53-german'
RATE = 44_100
CHUNK = 4096
CHANNELS = 1
FORMAT = pyaudio.paInt16
NUM_EMPTY_FRAMES_TO_STOP = 100

audio = pyaudio.PyAudio()

class Sample:

    def __init__(self, frames):
        self.frames = frames
    
    def to_numpy(self):
        return np.frombuffer(b''.join(self.frames), dtype=np.int16)

    def to_wav_file(self, file_path):
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

    def __add__(self, other):
        return self.__class__(self.frames + other.frames)

    def append(self, frame):
        self.frames.append(frame)

    def is_finished(self):
        if len(self.frames) > NUM_EMPTY_FRAMES_TO_STOP:
            return np.all(self.to_numpy()[-NUM_EMPTY_FRAMES_TO_STOP:] < 300)
        return False

    def is_empty(self):
        return np.all(self.to_numpy() < 500)

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
            frame = conn.recv(CHUNK)
            current_sample.append(frame)
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