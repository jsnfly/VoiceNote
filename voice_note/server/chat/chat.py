import asyncio
from typing import List, Union
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm import TokensPrompt

from server.base_server import BaseServer
from server.utils.streaming_connection import POLL_INTERVAL
from server.utils.message import Message

CHAT_MODEL = './models/chat/Qwen3-8B-FP8'
SYSTEM_PROMPT = """You are a helpful, smart and funny assistant talking directly to the user by leveraging
speech-to-text and text-to-speech. So keep your responses concise like in a real conversation and do not use any
spechial characters or emojis as they can not be expressed by the text-to-speech component.""".replace("\n", " ")
TTS_URI = 'ws://tts:12347'


class ChatServer(BaseServer):

    def __init__(self, host: str, port: int, tts_uri: Union[str, None] = None):
        super().__init__("chat", host, port)
        self.history = []
        if SYSTEM_PROMPT:
            self.history.append({'role': 'system', 'content': SYSTEM_PROMPT})

        engine_args = AsyncEngineArgs(model=CHAT_MODEL, tensor_parallel_size=1, gpu_memory_utilization=0.66,
                                      max_model_len=16384, max_num_seqs=1)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = None  # Initialized in `#serve_forever`

        if tts_uri is not None:
            self.connections = {'tts': tts_uri}

    async def serve_forever(self) -> None:
        """Initializes the tokenizer and then starts the server."""
        self.tokenizer = await self.engine.get_tokenizer()
        await super().serve_forever()

    def _recv_client_messages(self) -> List[Message.DataDict]:
        text_messages = []
        for msg in super()._recv_client_messages():
            if msg.get('action') == 'NEW CONVERSATION':
                self.history = self.history[:1] if SYSTEM_PROMPT else []
            else:
                text_messages.append(msg)
        return text_messages

    def _get_cutoff_idx(self, received: List[Message.DataDict]) -> int:
        return int(len(received) > 0)

    async def _run_workload(self, received: List[Message.DataDict]) -> None:
        history = self.history + [{'role': 'user', 'content': received[0]['text']}]
        prompt_token_ids = self.tokenizer.apply_chat_template(history, add_generation_prompt=True,
                                                              enable_thinking=False)
        request_id = received[0]['id']
        sampling_params = SamplingParams(max_tokens=8192)
        results_generator = self.engine.generate(prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
                                                 sampling_params=sampling_params, request_id=request_id)

        full_response = ""
        async for request_output in results_generator:
            generated_text = request_output.outputs[0].text
            new_text = generated_text[len(full_response):]
            full_response = generated_text

            is_finished = request_output.finished

            if new_text or is_finished:
                msg = {'status': 'FINISHED' if is_finished else 'GENERATING', 'text': new_text, 'id': request_id}

                if 'tts' in self.streams:
                    self.streams['tts'].send(msg)

                    # In case of a StreamReset, it is raised here and generation is interrupted. No need to use a cancellation
                    # Event.
                    # If TTS is used, the chat server should never send 'FINISHED' and only the TTS server should.
                    self.streams['client'].send(msg | {'status': 'GENERATING'})

                    [self.streams['client'].send(m) for m in self.streams['tts'].recv()]
                else:
                    self.streams['client'].send(msg)

        self.history = history  # Only add to permanent history if generation was not interrupted.
        self.history.append({'role': 'assistant', 'content': full_response})

        if 'tts' in self.streams:
            waiting_for_tts = True
            while waiting_for_tts:
                messages = self.streams['tts'].recv()
                for msg in messages:
                    self.streams['client'].send(msg)
                    if msg.get('status') == 'FINISHED':
                        waiting_for_tts = False
                if waiting_for_tts:
                    await asyncio.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    asyncio.run(ChatServer('0.0.0.0', '12346', TTS_URI).serve_forever())
