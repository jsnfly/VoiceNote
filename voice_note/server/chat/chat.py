import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from typing import Dict, List, Union

from server.base_server import BaseServer, ThreadExecutor
from server.utils.streaming_connection import POLL_INTERVAL, StreamingConnection
from server.utils.message import Message

CHAT_MODEL = './models/chat/gemma-3-4b-it'
SYSTEM_PROMPT = """Your name is George. Your are an intelligent, witty and pragmatic assistant. You are part of a
speech-to-speech pipeline, i.e. you can talk to the user directly. This means you should keep your answers concise,
like in a real conversation."""
TTS_URI = 'ws://localhost:12347'


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
            self.streams['tts'].send(msg)

            # In case of a StreamReset, it is raised here and generation is interrupted. No need to use a cancellation
            # Event.
            # If TTS is used, the chat server should never send 'FINISHED' and only the TTS server should.
            self.streams['client'].send(msg | {'status': 'GENERATING'})

            [self.streams['client'].send(msg) for msg in self.streams['tts'].recv()]
        else:
            self.streams['client'].send(msg)


class ChatServer(BaseServer):

    def __init__(self, host: str, port: int, tts_uri: Union[str, None] = None):
        super().__init__("chat", host, port)
        self.history = []
        if SYSTEM_PROMPT:
            self.history.append({'role': 'system', 'content': SYSTEM_PROMPT})
        self.generation = Generation()

        if tts_uri is not None:
            self.connections = {'tts': tts_uri}

    def _recv_client_messages(self) -> List[Message.DataDict]:
        text_messages = []
        for msg in super()._recv_client_messages():
            if msg.get('action') == 'NEW CHAT':
                self.history = self.history[:1] if SYSTEM_PROMPT else []
            else:
                text_messages.append(msg)
        return text_messages

    def _get_cutoff_idx(self, received: List[Message.DataDict]) -> int:
        return int(len(received) > 0)

    async def _run_workload(self, received: List[Message.DataDict]) -> None:
        history = self.history + [{'role': 'user', 'content': received[0]['text']}]
        inputs = self.generation.tokenizer.apply_chat_template(history, add_generation_prompt=True,
                                                               return_tensors='pt')
        result = await self.generation.run(inputs.cuda(), self.streams, received[0]['id'])

        self.history = history  # Only add to permanent history if generation was not interrupted.
        self.history.append({'role': 'assistant', 'content': result})

        waiting_for_tts = 'tts' in self.streams
        while waiting_for_tts:
            messages = self.streams['tts'].recv()
            for msg in messages:
                self.streams['client'].send(msg)
                waiting_for_tts = msg['status'] != 'FINISHED'
            await asyncio.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    asyncio.run(ChatServer('0.0.0.0', '12346', TTS_URI).serve_forever())
