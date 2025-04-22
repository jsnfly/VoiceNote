import asyncio
import re
import numpy as np
import torch
from transformers.generation.streamers import BaseStreamer
from typing import Dict, List

from server.base_server import BaseServer, ThreadExecutor
from server.tts.orpheus_model import OrpheusModel, DEVICE
from server.utils.message import Message
from server.utils.streaming_connection import StreamingConnection


class Generation(ThreadExecutor):
    def __init__(self):
        super().__init__()
        self.model = OrpheusModel()

    def blocking_fn(self, text: str, streams: Dict[str, StreamingConnection], id_: str, audio_config: dict) -> None:
        voice = "tara"
        prompt = (
            "<custom_token_3><|begin_of_text|>"
            f"{voice}: {text}"
            "<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"
        )
        model_inputs = self.model.tokenizer([prompt], return_tensors="pt").to(self.model.llm.device)
        self.model.llm.generate(
            **model_inputs, max_new_tokens=2048, temperature=0.6, top_p=0.8, repetition_penalty=1.3,
            eos_token_id=self.model.eos_token_id,
            streamer=Streamer(streams, id_, audio_config, self.model, chunk_size=35)
        )


class Streamer(BaseStreamer):

    def __init__(self, streams, id_, audio_config, model, chunk_size, overlap=14):
        self.streams = streams
        self.model = model

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.generated_tokens = []
        self.used_tokens = []

        self.skip_next=True
        self.base_msg = {
            "status": "GENERATING",
            "id": id_,
            "config": audio_config
        }

    def put(self, token_id):
        if self.skip_next:
            # The first generated "token" is actually the prompt and should be skipped.
            self.skip_next = False
            return

        token = self.model.tokenizer.decode(token_id)
        match = re.match(r"<custom_token_(\d+)>", token)
        if match is not None:
            self.generated_tokens.append(int(match.groups()[0]) - 10 - ((len(self.generated_tokens) % 7) * 4096))

        if len(self.generated_tokens) and (len(self.generated_tokens) % self.chunk_size == 0):
            self.generate_audio()

    def end(self):
        self.generate_audio()

    def generate_audio(self):
        overlap_tokens = self.used_tokens[-self.overlap:]
        new_tokens = self.generated_tokens[len(self.used_tokens):]
        self.used_tokens.extend(new_tokens)

        tokens = overlap_tokens + new_tokens
        tokens = [t for t in tokens if 0 <= t < 4096]
        tokens = torch.tensor(tokens, device=DEVICE)

        with torch.inference_mode():
            audio = self.model.convert_to_audio(tokens)
        audio = audio.detach().cpu().numpy()
        audio = audio[self.overlap // 7 * 2048:]  # 7 tokens give 2048 audio values

        bytes_ = (audio * 32767).astype(np.int16).tobytes()
        self.streams['client'].send(self.base_msg | {'audio': bytes_})


class TTSServer(BaseServer):
    def __init__(self, host: str, port: int):
        super().__init__("tts", host, port)
        self.generation = Generation()

        self.eos_rx = re.compile(r"(\.(?:\s|\Z))")
        self.audio_config = {
            'format': 8,  # 8 is pyaudio.paInt16.
            'channels': 1,
            'rate': 24_000
        }

    def _get_cutoff_idx(self, received: List[Message.DataDict]) -> int:
        idx = next((idx for idx in range(len(received)) if self._is_finished(received[:idx + 1])), -1) + 1
        return idx

    def _is_finished(self, messages: List[Message.DataDict]) -> bool:
        if messages[-1]['status'] == 'FINISHED':
            return True

        text = ''.join(msg['text'] for msg in messages)
        return len(text.split()) > 4 and self.eos_rx.search(text)

    async def _run_workload(self, received: List[Message.DataDict]) -> None:
        text = ''.join(msg['text'] for msg in received)
        print(f"{text}\n---")
        await self.generation.run(text, self.streams, received[0]['id'], self.audio_config)
        if received[-1]['status'] == 'FINISHED':
            self.streams['client'].send({'audio': b'', 'status': 'FINISHED', 'id': received[0]['id']})


if __name__ == '__main__':
    asyncio.run(TTSServer('0.0.0.0', '12347').serve_forever())
