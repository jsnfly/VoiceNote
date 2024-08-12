import asyncio
import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Union
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from server.base_server import BaseServer, ThreadExecutor
from server.utils.audio import AudioConfig
from server.utils.message import Message
from server.utils.sample import Sample
from server.utils.streaming_connection import POLL_INTERVAL

BASE_DIR = (Path(__file__).parent / '../../').resolve()
SAVE_DIR = BASE_DIR / 'outputs'
MODEL_DIR = BASE_DIR / 'models/whisper-medium'

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

    def blocking_fn(self, bytes_: bytes, audio_config: Dict, topic: str) -> Tuple[str, Path]:
        sample = Sample([bytes_], AudioConfig(**audio_config))
        sample.transcribe(self.model, self.processor)
        save_path = sample.save(SAVE_DIR / topic)
        print(sample.result)
        return sample.result, save_path


class STTServer(BaseServer):
    def __init__(self, host: str, port: int, chat_uri: Union[str, None] = None):
        super().__init__(host, port)
        self.transcription = Transcription()

        if chat_uri is not None:
            self.connections = {'chat': chat_uri}

    def _recv_client_messages(self) -> List[Message.DataDict]:
        audio_messages = []
        for msg in super()._recv_client_messages():
            action = msg.get('action')
            if action == 'DELETE':
                self.delete_entry(msg['save_path'])
            elif action == 'WRONG':
                self.add_to_metadata(msg['save_path'], {'transcription_error': True})
            elif action == 'NEW CHAT':
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

        bytes_ = b''.join([msg.get('audio', b'') for msg in messages])
        transcription, save_path = await self.transcription.run(bytes_, messages[0]['audio_config'],
                                                                messages[0]['topic'])

        result = {'status': 'FINISHED', 'text': transcription, 'save_path': str(save_path), 'id': messages[0]['id']}
        if 'chat' in self.streams and messages[0]['chat_mode']:
            await self.get_chat_response(result)
        else:
            self.streams['client'].send(result)

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

    async def get_chat_response(self, transcription_result: Message.DataDict) -> None:
        self.streams['chat'].send(transcription_result)
        while True:
            for msg in self.streams['chat'].recv():
                self.streams['client'].send(msg | {'save_path': transcription_result['save_path']})
                if msg["status"] == "FINISHED":
                    return
            await asyncio.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    asyncio.run(STTServer('0.0.0.0', '12345', CHAT_URI).serve_forever())
