import asyncio
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Union
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from server.base_server import BaseServer, ThreadExecutor
from websockets.server import WebSocketServerProtocol
from server.utils.audio import AudioConfig
from server.utils.conversation import Conversation
from server.utils.message import Message
from server.utils.misc import BASE_DIR
from server.utils.sample import Sample
from server.utils.streaming_connection import POLL_INTERVAL

SAVE_DIR = BASE_DIR / 'outputs'
MODEL_DIR = BASE_DIR / 'models/whisper-medium'
LANG = 'en'

DEVICE, DTYPE = ('cuda:0', torch.float16) if torch.cuda.is_available() else ('cpu', torch.float32)

CHAT_URI = 'ws://chat:12346'


class Transcription(ThreadExecutor):
    def __init__(self):
        super().__init__()
        self.processor = WhisperProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_DIR, use_safetensors=True, local_files_only=True, torch_dtype=DTYPE
        )
        self.model.to(DEVICE)

    def blocking_fn(self, sample: Sample) -> str:
        sample.transcribe(self.model, self.processor, LANG)
        return sample.result


class STTServer(BaseServer):
    def __init__(self, host: str, port: int, chat_uri: Union[str, None] = None):
        super().__init__("stt", host, port)
        self.transcription = Transcription()
        self.conversation: Conversation = None

        if chat_uri is not None:
            self.connections = {'chat': chat_uri}

    def _new_conversation(self) -> None:
        self.conversation = Conversation()

    async def handle_connection(self, client_connection: WebSocketServerProtocol) -> None:
        self._new_conversation()
        await super().handle_connection(client_connection)

    def _recv_client_messages(self) -> List[Message.DataDict]:
        audio_messages = []
        for msg in super()._recv_client_messages():
            action = msg.get('action')
            if action == 'DELETE':
                self.delete_entry(msg['save_path'])
            elif action == 'NEW CONVERSATION':
                self._new_conversation()
                if 'chat' in self.streams:
                    self.streams['chat'].reset(msg['id'])
                self.streams['chat'].send(msg)
            else:
                audio_messages.append(msg)
        return audio_messages

    def _get_cutoff_idx(self, received: List[Message.DataDict]) -> int:
        return next((idx for idx, msg in enumerate(received) if msg['status'] == 'FINISHED'), -1) + 1

    async def _run_workload(self, messages: List[Message.DataDict]) -> None:
        assert messages[0]['status'] == 'INITIALIZING'

        audio_config = AudioConfig(**messages[0]['audio_config'])
        audio_bytes = b''.join([msg.get('audio', b'') for msg in messages])
        sample = Sample(fragments=[audio_bytes], audio_config=audio_config)

        transcription = await self.transcription.run(sample)

        self.conversation.add_turn(
            user_text=transcription,
            user_audio_bytes=sample.get_audio_bytes(),
            user_audio_config=audio_config,
        )

        await self.get_chat_response(
            {'text': transcription, 'id': messages[0]['id']}
        )

    @staticmethod
    def delete_entry(save_path: str) -> None:
        save_path = Path(save_path)
        if not save_path.exists():
            return

        for file in save_path.iterdir():
            file.unlink()
        save_path.rmdir()
        print(f"Deleted {save_path}.")

    async def get_chat_response(self, transcription_result: Message.DataDict) -> None:
        self.streams['chat'].send(transcription_result)

        assistant_audio_config = None
        try:
            while True:
                for msg in self.streams['chat'].recv():
                    self.streams['client'].send(msg | {'save_path': self.conversation.get_save_path()})

                    if 'config' in msg and not assistant_audio_config:
                        assistant_audio_config = AudioConfig(**msg['config'])

                    self.conversation.update_assistant_response(
                        text_chunk=msg.get('text', ''),
                        audio_chunk=msg.get('audio', b''),
                    )

                    if msg.get("status") == "FINISHED":
                        return
                await asyncio.sleep(POLL_INTERVAL)
        finally:
            self.conversation.finalize_assistant_audio(assistant_audio_config)


if __name__ == '__main__':
    asyncio.run(STTServer('0.0.0.0', '12345', CHAT_URI).serve_forever())
