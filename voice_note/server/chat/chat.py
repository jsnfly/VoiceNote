import asyncio
import json
import re
from typing import List, Union
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm import TokensPrompt

from server.base_server import BaseServer
from server.chat.tools import TOOLS, MINIMAL_MODE_SYSTEM_PROMPT
from server.utils.streaming_connection import POLL_INTERVAL
from server.utils.message import Message

CHAT_MODEL = './models/chat/Qwen3-8B-FP8'
TTS_URI = 'ws://tts:12347'
DEFAULT_SYSTEM_PROMPT = """You are a helpful, smart and funny assistant talking directly to the user by leveraging
speech-to-text and text-to-speech. So keep your responses concise like in a real conversation and do not use any
spechial characters (including dashes, asteriks and so on) or emojis as they can not be expressed by the
text-to-speech component.""".replace("\n", " ")


class ChatServer(BaseServer):

    def __init__(self, host: str, port: int, tts_uri: Union[str, None] = None):
        super().__init__("chat", host, port)
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.reset_conversation()
        self.thinking_enabled = False

        engine_args = AsyncEngineArgs(model=CHAT_MODEL, tensor_parallel_size=1, gpu_memory_utilization=0.6,
                                      max_model_len=32768, max_num_seqs=1)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = None  # Initialized in `#serve_forever`

        if tts_uri is not None:
            self.connections = {'tts': tts_uri}

    async def serve_forever(self) -> None:
        """Initializes the tokenizer and then starts the server."""
        self.tokenizer = await self.engine.get_tokenizer()
        await super().serve_forever()

    def reset_conversation(self) -> None:
        self.history = [{'role': 'system', 'content': self.system_prompt}] if self.system_prompt else []

    def _recv_client_messages(self) -> List[Message.DataDict]:
        text_messages = []
        for msg in super()._recv_client_messages():
            if msg.get('action') == 'NEW CONVERSATION':
                self.reset_conversation()
            else:
                text_messages.append(msg)
        return text_messages

    def _get_cutoff_idx(self, received: List[Message.DataDict]) -> int:
        return int(len(received) > 0)

    async def _run_workload(self, received: List[Message.DataDict]) -> None:
        self.history.append({'role': 'user', 'content': received[0]['text']})
        print(self.history)
        prompt_token_ids = self.tokenizer.apply_chat_template(self.history, add_generation_prompt=True,
                                                              enable_thinking=self.thinking_enabled, tools=TOOLS)
        request_id = received[0]['id']
        sampling_params = SamplingParams(
            max_tokens=8192,
            temperature=0.6 if self.thinking_enabled else 0.7,
            top_p=0.95 if self.thinking_enabled else 0.8,
            top_k=20,
            min_p=0
        )
        results_generator = self.engine.generate(prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
                                                 sampling_params=sampling_params, request_id=request_id)
        processed_text = ''
        tool_calls = dict()
        is_finished = False
        try:
            idx = 0
            async for request_output in results_generator:
                idx += 1
                stream_text = ''
                generated_text = request_output.outputs[0].text
                if generated_text.count("<think>") > generated_text.count("</think>"):
                    # Thinking
                    if idx == 1 and self.system_prompt != MINIMAL_MODE_SYSTEM_PROMPT:
                        stream_text = 'Let me think about that.'
                elif generated_text.count("<tool_call>") > generated_text.count("</tool_call>"):
                    # Tool call
                    pass
                else:
                    for tool_call_str in re.findall(r"<tool_call>(.+?)</tool_call>", generated_text, re.DOTALL):
                        tool_call_str = tool_call_str.strip()
                        if tool_call_str not in tool_calls:
                            tool_call = json.loads(tool_call_str)
                            if tool_call['name'] == 'enable_thinking_mode':
                                enable = tool_call['arguments']['enable']
                                self.thinking_enabled = enable
                                tool_calls[tool_call_str] = {
                                    'name': tool_call['name'],
                                    'content': f"thinking mode {'enabled' if enable else 'diabled'}."
                                }
                                if enable:
                                    stream_text = 'Thinking mode enabled.'
                                else:
                                    stream_text = 'Thinking mode disabled.'
                            elif tool_call['name'] == 'enable_minimal_mode':
                                enable = tool_call['arguments']['enable']
                                self.system_prompt = MINIMAL_MODE_SYSTEM_PROMPT if enable else DEFAULT_SYSTEM_PROMPT
                                self.reset_conversation()
                                tool_calls[tool_call_str] = {
                                    'name': tool_call['name'],
                                }
                                if enable:
                                    stream_text = 'Starting a new conversation with minimal mode enabled.'
                                else:
                                    stream_text = 'Starting a new conversation with minimal mode disabled.'

                    new_text = generated_text[len(processed_text):]
                    processed_text = generated_text
                    new_content = new_text.split('</think>')[-1]
                    if self.history[-1]['role'] == 'assistant':
                        self.history[-1]['content'] += new_content
                    else:
                        new_content = new_content.lstrip("\n")
                        self.history.append({'role': 'assistant', 'content': new_content})

                    is_finished = request_output.outputs[0].finish_reason is not None
                    if '</tool_call>' in new_content:
                        stream_text += new_content.split('</tool_call>')[-1].lstrip("\n")
                    else:
                        stream_text += new_content

                if 'tts' in self.streams:
                    tts_msgs = self.streams['tts'].recv()
                    for tts_msg in tts_msgs:
                        self.streams['client'].send(tts_msg)

                if stream_text or is_finished:
                    self.streams['client'].send({
                        'status': 'FINISHED' if (is_finished and 'tts' not in self.streams) else 'GENERATING',
                        'text': stream_text,
                        'id': request_id
                    })
                    self.streams['tts'].send({
                        'status': 'FINISHED' if is_finished else 'GENERATING',
                        'text': stream_text,
                        'id': request_id
                    })
                await asyncio.sleep(POLL_INTERVAL)

            print(generated_text)
            for tool_call in tool_calls.values():
                if tool_call.get('content', ''):
                    self.history.append({'role': 'function', **tool_call})

            # After the loop, forward all remaining audio from TTS to the client
            if 'tts' in self.streams:
                waiting_for_tts = True
                while waiting_for_tts:
                    messages = self.streams['tts'].recv()
                    for msg in messages:
                        self.streams['client'].send(msg)
                        if msg.get('status') == 'FINISHED':
                            waiting_for_tts = False
                    await asyncio.sleep(POLL_INTERVAL)
                    if self.streams['tts'].closed and self.streams['tts'].received_q.empty():
                        break

        except asyncio.CancelledError:
            await self.engine.abort(request_id)
            raise

if __name__ == '__main__':
    asyncio.run(ChatServer('0.0.0.0', '12346', TTS_URI).serve_forever())
