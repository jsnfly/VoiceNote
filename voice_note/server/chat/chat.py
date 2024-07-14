import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from typing import Dict, List, Union

from server.base_server import BaseServer, ThreadExecutor
from server.utils.streaming_connection import POLL_INTERVAL, StreamingConnection
from server.utils.message import Message

CHAT_MODEL = './models/chat/openchat_3.5'
TTS_URI = 'ws://localhost:12347'


class Generation(ThreadExecutor):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, device_map="auto", torch_dtype="auto",
                                                          local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL, local_files_only=True)

    def blocking_fn(self, inputs, streams, id_):
        generation_config = self.model.generation_config
        generation_config.max_length = 2048
        streamer = Streamer(id_, streams, self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.model.generate(inputs, generation_config, streamer=streamer)
        return streamer.result


class Streamer(TextStreamer):
    def __init__(
        self,
        id_,
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

        if 'tts' in self.streams:
            self.streams['tts'].send(
                {'status': 'FINISHED' if stream_end else 'GENERATING', 'text': text, 'id': self.id}
            )

            # In case of a StreamReset, it is raised here and generation is interrupted. No need to use a cancellation
            # Event.
            self.streams['client'].send({'status': 'GENERATING', 'text': text, 'id': self.id})

            for msg in self.streams['tts'].recv():
                self.streams['client'].send(msg)
        else:
            self.streams['client'].send(
                {'status': 'FINISHED' if stream_end else 'GENERATING', 'text': text, 'id': self.id}
            )


class ChatServer(BaseServer):

    def __init__(self, host: str, port: int, tts_uri: Union[str, None] = None):
        super().__init__(host, port)
        self.system_prompt = 'Du bist ein lustiger KI-Assistent mit dem Namen Hubert.'
        self.history = [{'role': 'system', 'content': self.system_prompt}]
        self.generation = Generation()

        if tts_uri is not None:
            self.connections = {'tts': tts_uri}

    def _recv_client_messages(self) -> List[Message.DataDict]:
        text_messages = []
        for msg in super()._recv_client_messages():
            if msg.get('action') == 'NEW CHAT':
                self.history = self.history[:1]
            else:
                text_messages.append(msg)
        return text_messages

    def _get_cutoff_idx(self, received: List[Message.DataDict]) -> int:
        return int(len(received) > 0)

    async def _run_workload(self, received: List[Message.DataDict]) -> None:
        if 'tts' in self.streams:
            self.streams['tts'].reset(received[0]['id'])

        history = self.history + [{'role': 'user', 'content': received[0]['text']}]
        print(history)

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
