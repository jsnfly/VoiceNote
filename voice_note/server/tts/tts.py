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

sys.path.append('server/tts/xtts-runner/src')
from gpt import XTTSGPT
from tokenizer import TextTokenizer

TTS_MODEL = './models/XTTS-v2'
LANG = 'de'
SPEAKER = "Marcos Rudaski"


class Generation(ThreadExecutor):
    def __init__(self):
        super().__init__()
        self.text_tokenizer = TextTokenizer(str(Path(TTS_MODEL) / "vocab.json"))
        self.model = XTTSGPT(Path(TTS_MODEL) / "config.json")
        self.model.load(Path(TTS_MODEL) / "model.pth")
        self.model.cuda()
        self.model.set_speaker_embeddings(Path(TTS_MODEL) / "speakers_xtts.pth", SPEAKER)

    def blocking_fn(self, text: str, streams: Dict[str, StreamingConnection], id_: str, audio_config: dict) -> None:
        token_encoding = self.text_tokenizer.encode(text, LANG)
        input_ids = torch.tensor(
            token_encoding.ids + [self.model.config.gpt_start_audio_token], dtype=torch.int64
        ).unsqueeze(0)
        streamer = Streamer(streams, id_, audio_config, [], self.model.decoder, self.model.speaker_emb, chunk_size=20)
        self.model.generate(
            input_ids.cuda(),
            bos_token_id=self.model.config.gpt_start_audio_token,
            pad_token_id=self.model.config.gpt_stop_audio_token,
            eos_token_id=self.model.config.gpt_stop_audio_token,
            do_sample=True,
            top_p=0.85,
            top_k=50,
            temperature=0.75,
            num_return_sequences=1,
            num_beams=1,
            length_penalty=1.0,
            repetition_penalty=5.0,
            max_new_tokens=self.model.config.gpt_max_audio_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True,
            all_latents=streamer.all_latents,
            streamer=streamer
        )


class Streamer(BaseStreamer):

    def __init__(self, streams, id_, audio_config, all_latents, decoder, speaker_embedding, chunk_size):
        self.streams = streams
        self.all_latents = all_latents
        self.chunk_size = chunk_size
        self.decoder = decoder
        self.speaker_embedding = speaker_embedding.to(decoder.device)
        self.generated_tokens = []
        self.len_generated_audio_bytes = 0
        self.base_msg = {
            "status": "GENERATING",
            "id": id_,
            "config": audio_config
        }

    def put(self, token):
        self.generated_tokens.append(token)
        if len(self.generated_tokens) % self.chunk_size == 0:
            self.generate_audio()

    def end(self):
        self.generate_audio()

    def generate_audio(self):
        generated_latents = torch.cat(self.all_latents, dim=1)[:, -len(self.generated_tokens):]
        with torch.no_grad():
            generated_audio = self.decoder(generated_latents, g=self.speaker_embedding)

        bytes_ = generated_audio.squeeze().cpu().numpy().tobytes()
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
            'rate': self.generation.model.decoder.output_sample_rate
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
