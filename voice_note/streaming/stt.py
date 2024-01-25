import asyncio
import whisper
from pathlib import Path
from typing import Dict, Union

from base_server import BaseServer
from utils.audio import AudioConfig
from utils.message import Message
from utils.sample import Sample
from utils.streaming_connection import POLL_INTERVAL, StreamingConnection


SAVE_DIR = Path(__file__).parent.resolve() / 'outputs'
WHISPER_MODEL = 'medium'


class STTServer(BaseServer):
    def __init__(self, host: str, port: int, chat_uri: Union[str, None] = None):
        super().__init__(host, port)
        self.model = whisper.load_model(WHISPER_MODEL, device='cuda')
        self.decoding_options = whisper.DecodingOptions()

        self.received = []  # TODO: multiple connections
        self.chat_uri = chat_uri

    async def _handle_workload(self, connection: StreamingConnection) -> Message.DataDict:
        while True:
            end_idx = self._get_end_idx()
            if end_idx == -1:
                self.received += connection.recv()
                await asyncio.sleep(POLL_INTERVAL)
            else:
                messages = self.received[:end_idx + 1]
                self.received = self.received[end_idx + 1:]

                assert messages[0]['status'] == 'INITIALIZING'
                audio_config = messages[0]['audio_config']
                bytes_ = b''.join([msg.get('audio', b'') for msg in messages])

                transcription = await self.run_blocking_function_in_thread(self.transcribe, [bytes_, audio_config])
                result = {"status": "FINISHED", "text": transcription}
                if self.chat_uri is not None:
                    pass
                else:
                    connection.send(result)

    def transcribe(self, bytes_: bytes, audio_config: Dict):
        sample = Sample([bytes_], AudioConfig(**audio_config))
        sample.transcribe(self.model, self.decoding_options)
        # save_path = sample.save(SAVE_DIR / topic)
        print(sample.result.text)
        return sample.result.text  # , save_path

    # async def get_chat_response(self, msg: Message.DataDict) -> None:
    #     async with websockets.connect(self.chat_uri) as chat_websocket:
    #         await chat_websocket.send(Message(msg).encode())
    #         while True:
    #             response = Message.decode(await chat_websocket.recv())
    #             yield response
    #             if response["status"] == "FINISHED":
    #                 break

    def _get_end_idx(self) -> int:
        return next((idx for idx, msg in enumerate(self.received) if msg["status"] == "FINISHED"), -1)


if __name__ == '__main__':
    asyncio.run(STTServer('0.0.0.0', '12345', None).serve_forever())
