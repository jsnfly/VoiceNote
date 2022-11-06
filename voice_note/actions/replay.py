import wave
import re
from utils.pyaudio import audio
from server_config import SAVE_DIR

class Replay:
    def __init__(self):
        pass

    def trigger_condition(self, decoding_result):
        clean_text = re.sub("\W", "", decoding_result.text)
        return re.compile(r"replay", re.IGNORECASE).match(clean_text)

    def __call__(self, decoding_result):
        if not self.trigger_condition(decoding_result):
            return

        file_paths = list(SAVE_DIR.glob("*/sample.wav"))
        if file_paths:
            with wave.open(str(sorted(file_paths)[-1]), 'rb') as wf:
                stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                                    rate=wf.getframerate(), output=True)
                while len(frames := wf.readframes(1024)) > 0:
                    stream.write(frames)
                stream.stop_stream()
                stream.close()
