import asyncio
import re
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from typing import List

from server.base_server import BaseServer
from server.utils.message import Message

TTS_MODEL = './models/xtts/tts_models--multilingual--multi-dataset--xtts_v2'


class TTSServer(BaseServer):
    def __init__(self, host: str, port: int):
        super().__init__(host, port)
        config = XttsConfig()
        config.load_json(f"{TTS_MODEL}/config.json")
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=TTS_MODEL)
        self.model.cuda()

        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[f"{TTS_MODEL}/../sample.wav"]
        )

        self.eos_rx = re.compile(r"(\.(?:\s|\Z))")
        self.audio_config = {
            'format': 1,  # 1 is pyaudio.paFloat32.
            'channels': 1,
            'rate': self.model.config.audio.output_sample_rate
        }

    def _recv_client_messages(self) -> List[Message.DataDict]:
        return [msg for msg in self.streams['client'].recv() if msg['status'] != 'INITIALIZING']

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
        chunk_generator = self.model.inference_stream(text, "de", self.gpt_cond_latent, self.speaker_embedding)
        while True:
            # StopIteration does not play well with async code. Therefore handle generator exhaustion gracefully.
            chunk = await self.run_blocking_function_in_thread(lambda: next(chunk_generator, None), [])
            if chunk is None:
                break

            self.streams['client'].send({
                'audio': np.array(chunk.cpu(), dtype=np.float32).tobytes(),
                'status': 'GENERATING',
                'config': self.audio_config
            })
        if received[-1]['status'] == 'FINISHED':
            self.streams['client'].send({'audio': b'', 'status': 'FINISHED'})


if __name__ == '__main__':
    asyncio.run(TTSServer('0.0.0.0', '12347').serve_forever())
