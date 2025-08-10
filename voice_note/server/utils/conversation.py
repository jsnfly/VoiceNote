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
    def __init__(self):
        self.turns: List[Dict] = []
        self.assistant_audio_buffer = b""
        self.save_dir = Path('outputs')
        self.save_path = self.save_dir / time.strftime("%Y%m%d-%H%M%S")
        self.save_path.mkdir(parents=True, exist_ok=True)

    def add_turn(self, user_text: str, user_audio_bytes: bytes, user_audio_config: AudioConfig) -> None:
        turn_num = len(self.turns)
        user_audio_filename = f"user_audio_{turn_num}.wav"
        assistant_audio_filename = f"assistant_audio_{turn_num}.wav"

        # Reset the buffer for the new turn
        self.assistant_audio_buffer = b""

        self.turns.append({
            "turn": turn_num,
            "user": {
                "text": user_text,
                "audio_file": user_audio_filename
            },
            "assistant": {
                "text": "",
                "audio_file": assistant_audio_filename
            }
        })

        self._save_audio(user_audio_bytes, user_audio_config, user_audio_filename)
        self._save_json()

    def update_assistant_response(self, text_chunk: str, audio_chunk: bytes) -> None:
        if not self.turns:
            return

        last_turn = self.turns[-1]
        last_turn["assistant"]["text"] += text_chunk
        self.assistant_audio_buffer += audio_chunk
        self._save_json()

    def finalize_assistant_audio(self, audio_config: AudioConfig) -> None:
        if not self.turns or not audio_config:
            return

        last_turn = self.turns[-1]
        assistant_audio_filename = last_turn["assistant"]["audio_file"]

        self._save_audio(self.assistant_audio_buffer, audio_config, assistant_audio_filename)
        # No need to modify self.turns, as the buffer is separate.

    def _save_audio(self, audio_bytes: bytes, audio_config: AudioConfig, filename: str) -> None:
        if not audio_bytes or not audio_config:
            return

        filepath = self.save_path / filename
        audio_to_save = audio_bytes
        sample_width = audio_config.sample_size

        if audio_config.format == 1:  # pyaudio.paFloat32
            audio_to_save = _float_to_int16(audio_bytes)
            sample_width = 2  # 16-bit integer

        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(audio_config.channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(audio_config.rate)
            wf.writeframes(audio_to_save)

    def _save_json(self) -> None:
        filepath = self.save_path / "conversation.json"
        with open(filepath, 'w') as f:
            json.dump({
                "turns": self.turns
            }, f, indent=4)

    def get_save_path(self) -> str:
        return str(self.save_path)
