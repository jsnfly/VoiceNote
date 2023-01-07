import wave
import re
from .base import Action

class Replay(Action):

    def __init__(self, load_path):
        self.load_path = load_path

    def trigger_condition(self, decoding_result):
        clean_text = re.sub("\W", "", decoding_result.text)
        return re.compile(r"replay", re.IGNORECASE).match(clean_text)

    def run(self, _):
        file_paths = list(self.load_path.glob("*/sample.wav"))
        if file_paths:
            with wave.open(str(sorted(file_paths)[-1]), 'r') as wf:
                frames = wf.readframes(wf.getnframes())
            return {'audio': {'frames': frames, 'width': wf.getsampwidth(),'channels': wf.getnchannels(),
                              'rate': wf.getframerate()}}
