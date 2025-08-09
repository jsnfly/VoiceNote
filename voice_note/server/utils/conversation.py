import json
import wave
from pathlib import Path
import time
import numpy as np
from typing import Dict, List

from server.utils.audio import AudioConfig


def _float_to_int16(audio_bytes: bytes) -> bytes:
    """Converts a byte string of 32-bit floats to 16-bit integers."""
    float_array = np.frombuffer(audio_bytes, dtype=np.float32)
    int16_array = (float_array * 32767).astype(np.int16)
    return int16_array.tobytes()


class Conversation:
    def __init__(self, topic: str):
        self.topic = topic
        self.turns: List[Dict] = []
        self.save_dir = Path('outputs') / self.topic
        self.save_path = self.save_dir / time.strftime("%Y%m%d-%H%M%S")

    def add_turn(self, user_text: str, user_audio_bytes: bytes, user_audio_config: AudioConfig,
                 assistant_text: str, assistant_audio_bytes: bytes, assistant_audio_config: AudioConfig) -> None:
        turn_num = len(self.turns)
        user_audio_filename = f"user_audio_{turn_num}.wav"
        assistant_audio_filename = f"assistant_audio_{turn_num}.wav"

        self.turns.append({
            "turn": turn_num,
            "user": {
                "text": user_text,
                "audio_file": user_audio_filename
            },
            "assistant": {
                "text": assistant_text,
                "audio_file": assistant_audio_filename
            }
        })

        self._save_audio(user_audio_bytes, user_audio_config, user_audio_filename)
        self._save_audio(assistant_audio_bytes, assistant_audio_config, assistant_audio_filename)
        self._save_json()

    def _save_audio(self, audio_bytes: bytes, audio_config: AudioConfig, filename: str) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)
        filepath = self.save_path / filename

        audio_to_save = audio_bytes
        sample_width = audio_config.sample_size

        # pyaudio.paFloat32 has a value of 1
        if audio_config.format == 1:
            audio_to_save = _float_to_int16(audio_bytes)
            sample_width = 2  # 16-bit integer is 2 bytes

        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(audio_config.channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(audio_config.rate)
            wf.writeframes(audio_to_save)

    def _save_json(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)
        filepath = self.save_path / "conversation.json"
        with open(filepath, 'w') as f:
            json.dump({
                "topic": self.topic,
                "turns": self.turns
            }, f, indent=4)

    def get_save_path(self) -> str:
        return str(self.save_path)
