import asyncio
import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Union
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from server.base_server import BaseServer, ThreadExecutor, POLL_INTERVAL
from server.utils.audio import AudioConfig
from server.utils.message import DataDict
from server.utils.misc import BASE_DIR
from server.utils.sample import Sample
from server.utils.streaming_connection import StreamReset

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

    def blocking_fn(self, bytes_: bytes, audio_config: Dict, topic: str) -> Tuple[str, Path]:
        sample = Sample([bytes_], AudioConfig(**audio_config))
        sample.transcribe(self.model, self.processor, LANG)
        save_path = sample.save(SAVE_DIR / topic)
        print(f"Transcription: {sample.result}")
        return sample.result, save_path


class STTServer(BaseServer):
    def __init__(self, host: str, port: int, chat_uri: Union[str, None] = None):
        super().__init__("stt", host, port)
        self.transcription = Transcription()
        if chat_uri:
            self.connections['chat'] = chat_uri

    async def _main_loop(self) -> None:
        buffer: List[DataDict] = []
        workload_task = None

        while True:
            try:
                # Prioritize processing incoming messages
                new_messages = self.streams['client'].recv()
                for msg in new_messages:
                    # Handle session control messages immediately
                    if msg.get('action') == 'DELETE':
                        self.delete_entry(msg['save_path'])
                        continue
                    elif msg.get('action') == 'WRONG':
                        self.add_to_metadata(msg['save_path'], {'transcription_error': True})
                        continue
                    elif msg.get('action') == 'NEW CHAT':
                        if 'chat' in self.streams:
                            self.streams['chat'].reset(msg['id'])
                            self.streams['chat'].send(msg)
                        continue

                    # If a new session starts, clear buffer and cancel ongoing work
                    if buffer and msg['id'] != buffer[0]['id']:
                        print("New session started, clearing buffer.")
                        if workload_task and not workload_task.done():
                            workload_task.cancel()
                        buffer.clear()

                    buffer.append(msg)

                # Check if a full workload is ready to be processed
                if buffer and buffer[-1]['status'] == 'FINISHED':
                    # Reset downstream connections for the new workload
                    if 'chat' in self.streams:
                        self.streams['chat'].reset(buffer[0]['id'])

                    workload_task = asyncio.create_task(self._run_workload(buffer))
                    await workload_task
                    buffer = []

                await asyncio.sleep(POLL_INTERVAL)

            except StreamReset as e:
                print(f"Stream reset requested for id {e.id}. Clearing buffer.")
                if workload_task and not workload_task.done():
                    workload_task.cancel()
                buffer.clear()
                # Propagate reset to other streams
                for key, stream in self.streams.items():
                    if key != 'client':
                        stream.reset(e.id)
            except ConnectionError:
                print("Connection lost. Exiting main loop.")
                break

    async def _run_workload(self, messages: List[DataDict]) -> None:
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
        path = Path(save_path)
        if not path.exists():
            return
        for file in path.iterdir():
            file.unlink()
        path.rmdir()
        print(f"Deleted {path}.")

    @staticmethod
    def add_to_metadata(save_path: str, data: Dict) -> None:
        path = Path(save_path)
        if not path.exists():
            return
        metadata_path = path / 'metadata.json'
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        metadata.update(data)
        metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=4, ensure_ascii=False))

    async def get_chat_response(self, transcription_result: DataDict) -> None:
        self.streams['chat'].send(transcription_result)
        while True:
            try:
                for msg in self.streams['chat'].recv():
                    self.streams['client'].send(msg | {'save_path': transcription_result['save_path']})
                    if msg["status"] == "FINISHED":
                        return
                await asyncio.sleep(POLL_INTERVAL)
            except ConnectionError:
                print("Connection to chat server lost during response streaming.")
                break


if __name__ == '__main__':
    server = STTServer('0.0.0.0', 12345, CHAT_URI)
    asyncio.run(server.serve_forever())
