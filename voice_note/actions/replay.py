import wave
import re
from utils.pyaudio import audio
from server_config import SAVE_DIR

# TODO: refactor
def load_last_wavefile():
    file_paths = list(SAVE_DIR.glob("*/sample.wav"))
    if file_paths:
        all_frames = b''
        with wave.open(str(sorted(file_paths)[-1]), 'rb') as wf:
            while len(frames := wf.readframes(wf.getframerate())) > 0:
                # One iteration each second.
                all_frames += frames
            return {'frames': str(all_frames), 'width': wf.getsampwidth(),
                    'channels': wf.getnchannels(), 'rate': wf.getframerate()}

def trigger_condition(decoding_result):
    clean_text = re.sub("\W", "", decoding_result.text)
    return re.compile(r"replay", re.IGNORECASE).match(clean_text)
