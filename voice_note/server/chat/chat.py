import asyncio
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from server.base_server import BaseServer
from server.utils.streaming_connection import POLL_INTERVAL, StreamingConnection

CHAT_MODEL = "./models/chat/openchat_3.5"


class Streamer(TextStreamer):
    def __init__(
        self, connection: StreamingConnection, tokenizer: "AutoTokenizer", skip_prompt: bool = False, **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.connection = connection

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.connection.send({'status': 'FINISHED' if stream_end else 'GENERATING', 'text': text})


class ChatServer(BaseServer):

    def __init__(self, host: str, port: int):
        super().__init__(host, port)
        self.tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, device_map="auto", torch_dtype="auto",
                                                          local_files_only=True)
        self.system_prompt = 'Du bist ein lustiger KI-Assistent mit dem Namen Hubert.'

    async def _handle_workload(self) -> None:
        received = []

        while True:
            try:
                while len(received) == 0:
                    received += self.connections['client'].recv()
                    await asyncio.sleep(POLL_INTERVAL)

                # TODO: history
                messages = [
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': received.pop(0)['text']}
                ]

                generation_config = self.model.generation_config
                generation_config.max_length = 2048

                inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
                streamer = Streamer(self.connections['client'], self.tokenizer, skip_prompt=True,
                                    skip_special_tokens=True)
                await self.run_blocking_function_in_thread(
                    partial(self.model.generate, inputs.cuda(), generation_config, streamer=streamer)
                )
            except ConnectionError:
                break


if __name__ == '__main__':
    asyncio.run(ChatServer('0.0.0.0', '12346').serve_forever())
