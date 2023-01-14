import wave
import re
from pathlib import Path
from actions.action import Action

class Records(Action):

    def __init__(self, load_path):
        self.load_path = Path(load_path)

    def trigger_condition(self, decoding_result):
        clean_text = self.remove_non_word_characters(decoding_result.text)
        return re.compile(r"|".join(self.TRIGGER_WORDS), re.IGNORECASE).match(clean_text)

    def get_last_record(self):
        record_dirs = list(self.load_path.iterdir())
        return sorted(record_dirs)[-1] if record_dirs else None

    @staticmethod
    def remove_non_word_characters(text):
        return re.sub("\W", "", text)
