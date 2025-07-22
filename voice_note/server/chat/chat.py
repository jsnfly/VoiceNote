import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from typing import Dict, Union

from server.base_server import BaseServer, ThreadExecutor, POLL_INTERVAL
from server.utils.streaming_connection import StreamingConnection, StreamReset
from server.utils.message import DataDict

CHAT_MODEL = './models/chat/gemma-3-4b-it'
SYSTEM_PROMPT = """Your name is George. Your are an intelligent, witty and pragmatic assistant. You are part of a
speech-to-speech pipeline, i.e. you can talk to the user directly. This means you should keep your answers concise,
like in a real conversation."""
TTS_URI = 'ws://tts:12347'


class Generation(ThreadExecutor):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, local_files_only=True, torch_dtype="auto")
        self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL, local_files_only=True)

    def blocking_fn(self, inputs: torch.Tensor, streams: Dict[str, StreamingConnection], id_: str) -> str:
        generation_config = self.model.generation_config
        generation_config.max_length = 8192
        streamer = Streamer(id_, streams, self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.model.generate(inputs, generation_config, streamer=streamer)
        return streamer.result


class Streamer(TextStreamer):
    def __init__(
        self,
        id_: str,
        streams: Dict[str, StreamingConnection],
        tokenizer: "AutoTokenizer",
        skip_prompt: bool = False,
        **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.id = id_
        self.streams = streams
        self.result = ''

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.result += text
        msg = {'status': 'FINISHED' if stream_end else 'GENERATING', 'text': text, 'id': self.id}

        if 'tts' in self.streams:
            try:
                self.streams['tts'].send(msg)
                # If TTS is used, the chat server should never send 'FINISHED' and only the TTS server should.
                self.streams['client'].send(msg | {'status': 'GENERATING'})
                # Forward TTS audio back to the client
                for tts_msg in self.streams['tts'].recv():
                    self.streams['client'].send(tts_msg)
            except (ConnectionError, StreamReset) as e:
                print(f"Error communicating with TTS service: {e}")
                # If TTS fails, send the final text directly to the client
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
        client_stream = self.streams['client']
        workload_task = None

        while True:
            try:
                msg = await client_stream.received_q.get()

                if msg.get('action') == 'NEW CHAT':
                    print("New chat session started.")
                    self.history = self.history[:1] if SYSTEM_PROMPT else []
                    if workload_task and not workload_task.done():
                        workload_task.cancel()
                    # Reset tts connection as well
                    if 'tts' in self.streams:
                        self.streams['tts'].reset(msg['id'])
                    continue

                # If a new transcription comes in, cancel any ongoing generation
                if workload_task and not workload_task.done():
                    print("New transcription received, cancelling previous generation.")
                    workload_task.cancel()

                # Reset downstream connections for the new workload
                if 'tts' in self.streams:
                    self.streams['tts'].reset(msg['id'])

                workload_task = asyncio.create_task(self._run_workload(msg))
                await workload_task

            except StreamReset as e:
                print(f"Stream reset requested for id {e.id}.")
                if workload_task and not workload_task.done():
                    workload_task.cancel()
            except ConnectionError:
                print("Connection lost. Exiting main loop.")
                break

    async def _run_workload(self, msg: DataDict) -> None:
        history = self.history + [{'role': 'user', 'content': msg['text']}]
        inputs = self.generation.tokenizer.apply_chat_template(
            history, add_generation_prompt=True, return_tensors='pt'
        )

        try:
            result = await self.generation.run(inputs.cuda(), self.streams, msg['id'])
            # Only add to permanent history if generation was not interrupted.
            self.history = history
            self.history.append({'role': 'assistant', 'content': result})

            # If using TTS, wait for the final audio chunk to be sent.
            if 'tts' in self.streams:
                await self._wait_for_tts_completion()

        except asyncio.CancelledError:
            print(f"Workload for id {msg['id']} was cancelled.")
            raise

    async def _wait_for_tts_completion(self) -> None:
        """
        Waits for the final 'FINISHED' status from the TTS server.
        """
        tts_stream = self.streams.get('tts')
        if not tts_stream:
            return

        while True:
            try:
                messages = tts_stream.recv()
                for msg in messages:
                    self.streams['client'].send(msg)
                    if msg.get('status') == 'FINISHED':
                        return
                await asyncio.sleep(POLL_INTERVAL)
            except ConnectionError:
                print("Connection to TTS lost while waiting for completion.")
                break


if __name__ == '__main__':
    server = ChatServer('0.0.0.0', 12346, TTS_URI)
    asyncio.run(server.serve_forever())
