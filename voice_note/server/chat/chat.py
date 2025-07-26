import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from typing import Dict, Union
import threading

from server.base_server import BaseServer, ThreadExecutor, POLL_INTERVAL
from server.utils.streaming_connection import StreamingConnection, StreamReset
from server.utils.message import DataDict

CHAT_MODEL = './models/chat/gemma-3-4b-it'
SYSTEM_PROMPT = """Your name is George. Your are an intelligent, witty and pragmatic assistant. You are part of a
speech-to-speech pipeline, i.e. you can talk to the user directly. This means you should keep your answers concise,
like in a real conversation."""
TTS_URI = 'ws://tts:12347'


class GenerationCancelled(Exception):
    """Custom exception to signal that a generation task was cancelled."""
    pass


class Generation(ThreadExecutor):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, local_files_only=True, torch_dtype="auto")
        self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL, local_files_only=True)

    def blocking_fn(self, inputs: torch.Tensor, streams: Dict[str, StreamingConnection], id_: str) -> str:
        generation_config = self.model.generation_config
        generation_config.max_length = 8192
        streamer = Streamer(id_, streams, self.tokenizer, self.cancel_event, skip_prompt=True, skip_special_tokens=True)
        try:
            self.model.generate(inputs, generation_config, streamer=streamer)
        except GenerationCancelled:
            print(f"Generation for id {id_} was cancelled in thread.")
        return streamer.result


class Streamer(TextStreamer):
    def __init__(
        self,
        id_: str,
        streams: Dict[str, StreamingConnection],
        tokenizer: "AutoTokenizer",
        cancel_event: threading.Event,
        skip_prompt: bool = False,
        **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.id = id_
        self.streams = streams
        self.cancel_event = cancel_event
        self.result = ''

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if self.cancel_event.is_set():
            raise GenerationCancelled()

        self.result += text
        msg = {'status': 'FINISHED' if stream_end else 'GENERATING', 'text': text, 'id': self.id}

        if 'tts' in self.streams:
            try:
                self.streams['tts'].send(msg)
                self.streams['client'].send(msg | {'status': 'GENERATING'})
            except (ConnectionError, StreamReset):
                self.streams['client'].send(msg | {'status': 'FINISHED'})
        else:
            self.streams['client'].send(msg)


class ChatServer(BaseServer):
    def __init__(self, host: str, port: int, tts_uri: Union[str, None] = None):
        super().__init__("chat", host, port)
        self.history = []
        if SYSTEM_PROMPT:
            self.history.append({'role': 'system', 'content': SYSTEM_PROMPT})
        self.generation = Generation()
        if tts_uri:
            self.connections['tts'] = tts_uri

    async def _main_loop(self) -> None:
        workload_task = None
        
        forwarding_task = None
        if 'tts' in self.streams:
            forwarding_task = asyncio.create_task(self._forward_tts_audio())

        try:
            while True:
                msg = await self.streams['client'].received_q.get()

                if workload_task and not workload_task.done():
                    workload_task.cancel()

                if msg.get('action') == 'NEW CHAT':
                    self.history = self.history[:1] if SYSTEM_PROMPT else []
                    if 'tts' in self.streams:
                        self.streams['tts'].reset(msg['id'])
                    continue

                if 'tts' in self.streams:
                    self.streams['tts'].reset(msg['id'])

                workload_task = asyncio.create_task(self._run_workload(msg))

        except (ConnectionError, asyncio.CancelledError):
            print("Connection lost or task cancelled. Exiting main loop.")
        finally:
            if workload_task and not workload_task.done():
                workload_task.cancel()
            if forwarding_task and not forwarding_task.done():
                forwarding_task.cancel()

    async def _run_workload(self, msg: DataDict) -> None:
        history = self.history + [{'role': 'user', 'content': msg['text']}]
        inputs = self.generation.tokenizer.apply_chat_template(
            history, add_generation_prompt=True, return_tensors='pt'
        )
        
        try:
            result = await self.generation.run(inputs.cuda(), self.streams, msg['id'])
            if not self.generation.cancel_event.is_set():
                self.history = history
                self.history.append({'role': 'assistant', 'content': result})

        except asyncio.CancelledError:
            print(f"Workload for id {msg['id']} was cancelled.")
        except Exception as e:
            print(f"An error occurred in workload for id {msg['id']}: {e}")

    async def _forward_tts_audio(self) -> None:
        tts_stream = self.streams.get('tts')
        client_stream = self.streams.get('client')
        if not tts_stream or not client_stream:
            return

        while True:
            try:
                messages = tts_stream.recv()
                for msg in messages:
                    client_stream.send(msg)
                await asyncio.sleep(POLL_INTERVAL)
            except (ConnectionError, asyncio.CancelledError):
                print("TTS forwarding task stopping.")
                break


if __name__ == '__main__':
    server = ChatServer('0.0.0.0', 12346, TTS_URI)
    asyncio.run(server.serve_forever())

