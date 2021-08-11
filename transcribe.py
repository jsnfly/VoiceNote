import pyaudio
import time
import torch
import numpy as np
import wave
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WAVE_OUTPUT_FILENAME = "output.wav"
MODEL = 'jonatasgrosman/wav2vec2-large-xlsr-53-german'
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
            return self.to_numpy()[-NUM_EMPTY_FRAMES_TO_STOP:].mean() == 0
        return False

    def is_empty(self):
        return self.to_numpy().mean() == 0

current_sample = Sample([])
processor = Wav2Vec2Processor.from_pretrained(MODEL)
model = Wav2Vec2ForCTC.from_pretrained(MODEL)

def save_audio_and_prediction(save_path, sample, prediction):
    save_path = Path(save_path) / time.strftime("%Y%m%d-%H%M%S")
    save_path.mkdir(parents=True)
    with open(save_path / 'prediction.txt', 'w') as f:
        f.write(prediction)
    sample.to_wav_file(str(save_path / 'sample.wav'))

def callback(in_data, frame_count, time_info, status):
    global current_sample

    message = pyaudio.paContinue
    current_sample.append(in_data)
    if current_sample.is_finished():
        if not current_sample.is_empty():
            inputs = processor(current_sample.to_numpy(), sampling_rate=16_000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            prediction = processor.batch_decode(predicted_ids)[0]
            print("Prediction:", prediction)
            save_audio_and_prediction('outputs', current_sample, prediction)
            if prediction in ['STOP', 'STOPP']:
                message = pyaudio.paComplete
        current_sample = Sample([])
    return (in_data, message)

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK, 
    stream_callback=callback
)

print("* recording")

while stream.is_active():
    time.sleep(0.01)

stream.stop_stream()
stream.close()
audio.terminate()


