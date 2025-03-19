import asyncio
import re
import sys
import torch
from pathlib import Path
from transformers.generation.streamers import BaseStreamer
from typing import Dict, List

from server.base_server import BaseServer, ThreadExecutor
from server.utils.message import Message
from server.utils.streaming_connection import StreamingConnection

sys.path.append('server/tts/Spark-TTS')
from cli.SparkTTS import SparkTTS

TTS_MODEL = './models/Spark-TTS-0.5B'
SPEECH_CONFIG = {
    'gender': 'male',
    'pitch': 'high',
    'speed': 'high'
}


class Generation(ThreadExecutor):
    def __init__(self):
        super().__init__()
        self.model = SparkTTS(TTS_MODEL, torch.device("cuda"))

    def blocking_fn(self, text: str, streams: Dict[str, StreamingConnection], id_: str, audio_config: dict) -> None:
        prompt = self.model.process_prompt_control(text=text, **SPEECH_CONFIG)
        model_inputs = self.model.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        streamer = Streamer(streams, id_, audio_config, self.model, chunk_size=30)
        self.model.model.generate(
            **model_inputs,
            max_new_tokens=3000,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            streamer=streamer
        )


class Streamer(BaseStreamer):

    def __init__(self, streams, id_, audio_config, model, chunk_size):
        self.streams = streams
        self.chunk_size = chunk_size
        self.model = model
        self.generated_tokens = []
        self.len_generated_audio_bytes = 0
        self.base_msg = {
            "status": "GENERATING",
            "id": id_,
            "config": audio_config
        }
        self.skip_next=True

    def put(self, token_id):
        if self.skip_next:
            # The first generated "token" is actually the prompt and should be skipped.
            self.skip_next = False
            return

        self.generated_tokens.extend(self.model.tokenizer.convert_ids_to_tokens(token_id))
        semantic_tokens = [token for token in self.generated_tokens if "semantic" in token]
        if len(semantic_tokens) and (len(semantic_tokens) % self.chunk_size == 0):
            self.generate_audio()

    def end(self):
        self.generate_audio()

    def generate_audio(self):
        predicts = ''.join(self.generated_tokens)

        # Extract semantic token IDs from the generated text
        pred_semantic_ids = (
            torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)])
            .long()
            .unsqueeze(0)
        )

        global_token_ids = (
            torch.tensor([int(token) for token in re.findall(r"bicodec_global_(\d+)", predicts)])
            .long()
            .unsqueeze(0)
            .unsqueeze(0)
        )

        with torch.no_grad():
            # Convert semantic tokens back to waveform
            wav = self.model.audio_tokenizer.detokenize(
                global_token_ids.to(self.model.device).squeeze(0), pred_semantic_ids.to(self.model.device),
            )

        bytes_ = wav.tobytes()
        self.streams['client'].send(self.base_msg | {'audio': bytes_[self.len_generated_audio_bytes:]})
        self.len_generated_audio_bytes = len(bytes_)


class TTSServer(BaseServer):
    def __init__(self, host: str, port: int):
        super().__init__("tts", host, port)
        self.generation = Generation()

        self.eos_rx = re.compile(r"(\.(?:\s|\Z))")
        self.audio_config = {
            'format': 1,  # 1 is pyaudio.paFloat32.
            'channels': 1,
            'rate': self.generation.model.sample_rate
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
