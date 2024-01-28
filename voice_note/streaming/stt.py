import asyncio
import json
import whisper
from pathlib import Path
from typing import Dict, Tuple, Union

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
            try:
                end_idx = self._get_end_idx()
                if end_idx == -1:
                    new_messages = connection.recv()
                    for msg in new_messages:
                        action = msg.get('action')
                        if action == 'DELETE':
                            self.delete_entry(msg['save_path'])
                        elif action == 'WRONG':
                            self.add_to_metadata(msg['save_path'], {'transcription_error': True})
                    self.received += new_messages
                    await asyncio.sleep(POLL_INTERVAL)
                else:
                    messages = self.received[:end_idx + 1]
                    self.received = self.received[end_idx + 1:]

                    assert messages[0]['status'] == 'INITIALIZING'
                    bytes_ = b''.join([msg.get('audio', b'') for msg in messages])

                    transcription, save_path = await self.run_blocking_function_in_thread(
                        self.transcribe, [bytes_, messages[0]['audio_config'], messages[0]['topic']]
                    )
                    result = {'status': 'FINISHED', 'text': transcription, 'save_path': str(save_path)}
                    if self.chat_uri is not None:
                        pass
                    else:
                        connection.send(result)
            except ConnectionError:
                break

    def _get_end_idx(self) -> int:
        return next((idx for idx, msg in enumerate(self.received) if msg['status'] == 'FINISHED'), -1)

    @staticmethod
    def delete_entry(save_path: str) -> None:
        save_path = Path(save_path)
        if not save_path.exists():
            return

        for file in save_path.iterdir():
            file.unlink()
        save_path.rmdir()
        print(f"Deleted {save_path}.")

    @staticmethod
    def add_to_metadata(save_path: str, data: Dict) -> None:
        save_path = Path(save_path)
        if not save_path.exists():
            return

        metadata_path = save_path / 'metadata.json'
        if metadata_path.exists():
            with metadata_path.open() as f:
                metadata = json.load(f)
        else:
            metadata = {}

        metadata = metadata | data
        with metadata_path.open('w') as f:
            json.dump(metadata, f, sort_keys=True, indent=4, ensure_ascii=False)

    def transcribe(self, bytes_: bytes, audio_config: Dict, topic: str) -> Tuple[str, Path]:
        sample = Sample([bytes_], AudioConfig(**audio_config))
        sample.transcribe(self.model, self.decoding_options)
        save_path = sample.save(SAVE_DIR / topic)
        print(sample.result.text)
        return sample.result.text, save_path

    # async def get_chat_response(self, msg: Message.DataDict) -> None:
    #     async with websockets.connect(self.chat_uri) as chat_websocket:
    #         await chat_websocket.send(Message(msg).encode())
    #         while True:
    #             response = Message.decode(await chat_websocket.recv())
    #             yield response
    #             if response["status"] == "FINISHED":
    #                 break


if __name__ == '__main__':
    asyncio.run(STTServer('0.0.0.0', '12345', None).serve_forever())
