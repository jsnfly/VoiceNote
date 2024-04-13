import asyncio
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from typing import Dict, List, Union

from server.base_server import BaseServer
from server.utils.streaming_connection import POLL_INTERVAL, StreamingConnection
from server.utils.message import Message

CHAT_MODEL = './models/chat/openchat_3.5'
TTS_URI = 'ws://tts:12347'


class Streamer(TextStreamer):
    def __init__(
        self, streams: Dict[str, StreamingConnection],
        tokenizer: "AutoTokenizer",
        skip_prompt: bool = False,
        **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.streams = streams
        self.result = ''

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.result += text

        if 'tts' in self.streams:
            self.streams['tts'].send({'status': 'FINISHED' if stream_end else 'GENERATING', 'text': text})
            self.streams['client'].send({'status': 'GENERATING', 'text': text})
            for msg in self.streams['tts'].recv():
                self.streams['client'].send(msg)
        else:
            self.streams['client'].send({'status': 'FINISHED' if stream_end else 'GENERATING', 'text': text})


class ChatServer(BaseServer):

    def __init__(self, host: str, port: int, tts_uri: Union[str, None] = None):
        super().__init__(host, port)
        self.tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, device_map="auto", torch_dtype="auto",
                                                          local_files_only=True)
        self.system_prompt = 'Du bist ein lustiger KI-Assistent mit dem Namen Hubert.'
        self.history = [{'role': 'system', 'content': self.system_prompt}]

        if tts_uri is not None:
            self.connections = {'tts': tts_uri}

    def _recv_client_messages(self) -> List[Message.DataDict]:
        text_messages = []
        for msg in super()._recv_client_messages():
            if msg.get('action') == 'NEW CHAT':
                self.history = self.history[:1]
            elif msg.get('status') == 'INITIALIZING':
                continue
            else:
                text_messages.append(msg)
        return text_messages

    def _get_cutoff_idx(self, received: List[Message.DataDict]) -> int:
        return int(len(received) > 0)

    async def _run_workload(self, received: List[Message.DataDict]) -> None:
        self.history.append({'role': 'user', 'content': received[0]['text']})

        generation_config = self.model.generation_config
        generation_config.max_length = 2048

        inputs = self.tokenizer.apply_chat_template(self.history, add_generation_prompt=True, return_tensors='pt')
        streamer = Streamer(self.streams, self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.streams['client'].send({'status': 'INITIALIZING'})
        if 'tts' in self.streams:
            self.streams['tts'].send({'status': 'INITIALIZING'})
        await self.run_blocking_function_in_thread(
            partial(self.model.generate, inputs.cuda(), generation_config, streamer=streamer)
        )
        self.history.append({'role': 'assistant', 'content': streamer.result})

        waiting_for_tts = 'tts' in self.streams
        while waiting_for_tts:
            messages = self.streams['tts'].recv()
            for msg in messages:
                self.streams['client'].send(msg)
                waiting_for_tts = msg['status'] != 'FINISHED'
            await asyncio.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    asyncio.run(ChatServer('0.0.0.0', '12346', TTS_URI).serve_forever())
