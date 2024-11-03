import asyncio
import re
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from typing import List

from server.base_server import BaseServer, ThreadExecutor
from server.utils.message import Message
from server.utils.streaming_connection import POLL_INTERVAL

TTS_MODEL = './models/XTTS-v2'
LANG = 'de'
SAMPLE = f"{TTS_MODEL}/samples/{LANG}_sample.wav"


class Generation(ThreadExecutor):
    def __init__(self):
        super().__init__()
        config = XttsConfig()
        config.load_json(f"{TTS_MODEL}/config.json")
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=TTS_MODEL)
        self.model.cuda()

        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(audio_path=[SAMPLE])
        self.sample_rate = self.model.config.audio.output_sample_rate

    def blocking_fn(self):
        try:
            # StopIteration does not play well with async code. Therefore handle generator exhaustion gracefully.
            x = next(self.inference_stream, None)
            return x
        except ValueError:
            # TODO: Ideally this should not be necessary. It seems there can be multiple threads accessing
            #  `inference_stream` at the same time, but not sure why.
            return 'DUMMY'

    def initialize_stream(self, text):
        self.inference_stream = self.model.inference_stream(text, LANG, self.gpt_cond_latent, self.speaker_embedding)


class TTSServer(BaseServer):
    def __init__(self, host: str, port: int):
        super().__init__("tts", host, port)
        self.generation = Generation()

        self.eos_rx = re.compile(r"(\.(?:\s|\Z))")
        self.audio_config = {
            'format': 1,  # 1 is pyaudio.paFloat32.
            'channels': 1,
            'rate': self.generation.sample_rate
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
        self.generation.initialize_stream(text)
        while True:
            chunk = await self.generation.run()
            if chunk is None:
                break
            elif chunk == "DUMMY":
                await asyncio.sleep(POLL_INTERVAL)
                continue

            self.streams['client'].send({
                'audio': np.array(chunk.cpu(), dtype=np.float32).tobytes(),
                'status': 'GENERATING',
                'config': self.audio_config,
                'id': received[0]['id']
            })
        if received[-1]['status'] == 'FINISHED':
            self.streams['client'].send({'audio': b'', 'status': 'FINISHED', 'id': received[0]['id']})


if __name__ == '__main__':
    asyncio.run(TTSServer('0.0.0.0', '12347').serve_forever())
