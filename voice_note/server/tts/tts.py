import asyncio
import re
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from server.base_server import BaseServer
from server.utils.streaming_connection import POLL_INTERVAL

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

    async def _run_workload(self) -> None:
        text = ''
        finished_receiving = False

        self.streams['client'].send({'status': 'INITIALIZING'})
        while True:
            new_messages = self.streams['client'].recv()
            if new_messages and new_messages[-1]['status'] == 'FINISHED':
                finished_receiving = True

            text = ''.join([text, *[msg.get('text', '') for msg in new_messages]])
            parts = self.eos_rx.split(text)
            if len(parts) > 1 and len(parts[0].split()) > 4:
                current_sentence = parts[0] + parts[1]
                text = ''.join(parts[2:])
            elif finished_receiving:
                current_sentence = text
                text = ''
            else:
                current_sentence = ''
                await asyncio.sleep(POLL_INTERVAL)

            if current_sentence:
                print(current_sentence)
                chunk_generator = self.model.inference_stream(
                    current_sentence, "de", self.gpt_cond_latent, self.speaker_embedding
                )
                while True:
                    # StopIteration does not play well with async code. Therefore handle generator exhaustion
                    # gracefully
                    chunk = await self.run_blocking_function_in_thread(lambda: next(chunk_generator, None), [])

                    if chunk is None:
                        break

                    self.streams['client'].send({
                        'audio': np.array(chunk.cpu(), dtype=np.float32).tobytes(),
                        'status': 'GENERATING',
                        'config': self.audio_config
                    })
            if finished_receiving and text == '':
                self.streams['client'].send({'audio': b'', 'status': 'FINISHED'})
                finished_receiving = False


if __name__ == '__main__':
    asyncio.run(TTSServer('0.0.0.0', '12347').serve_forever())
