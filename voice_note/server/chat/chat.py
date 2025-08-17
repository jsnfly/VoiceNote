import asyncio
import json
from typing import List, Union
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm import TokensPrompt

from server.base_server import BaseServer
from server.utils.streaming_connection import POLL_INTERVAL
from server.utils.message import Message

CHAT_MODEL = './models/chat/Qwen3-8B-FP8'
TOOL_PROMPT = """You have access to the following tools. To use a tool, you must respond *only* with a JSON object
inside a <tool_call> block. The JSON should have "name" and "parameters" keys.
Example: <tool_call>{"name": "enable_thinking", "parameters": {"enable": true}}</tool_call>
Available tools: - Tool `enable_thinking`: Enables or disables the "thinking mode" for generating responses.
Parameters: `{"enable": <boolean>}`""".replace("\n", " ")
SYSTEM_PROMPT = f"""You are a helpful, smart and funny assistant talking directly to the user by leveraging
speech-to-text and text-to-speech. So keep your responses concise like in a real conversation and do not use any
spechial characters (including dashes, asteriks and so on) or emojis as they can not be expressed by the
text-to-speech component. {TOOL_PROMPT}""".replace("\n", " ")
TTS_URI = 'ws://tts:12347'


class ChatServer(BaseServer):

    def __init__(self, host: str, port: int, tts_uri: Union[str, None] = None):
        super().__init__("chat", host, port)
        self.history = []
        if SYSTEM_PROMPT:
            self.history.append({'role': 'system', 'content': SYSTEM_PROMPT})

        self.thinking_enabled = False
        self.tools = {
            "enable_thinking": self._tool_enable_thinking
        }

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

    def _tool_enable_thinking(self, enable: bool) -> str:
        self.thinking_enabled = bool(enable)
        mode = "enabled" if self.thinking_enabled else "disabled"
        return f"Okay, I have {mode} thinking mode."

    def _execute_tool_call(self, tool_call_str: str) -> str:
        try:
            tool_call = json.loads(tool_call_str)
            tool_name = tool_call.get("name")
            tool_params = tool_call.get("parameters", {})

            if tool_name in self.tools:
                tool_func = self.tools[tool_name]
                return tool_func(**tool_params)
            else:
                return f"Sorry, I don't know the tool '{tool_name}'."
        except json.JSONDecodeError:
            return "Sorry, I received an invalid tool call format."
        except Exception as e:
            return f"An error occurred while executing the tool: {e}"

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
        user_prompt = {'role': 'user', 'content': received[0]['text']}
        history = self.history + [user_prompt]
        self.history = history + [{'role': 'assistant', 'content': ''}]

        prompt_token_ids = self.tokenizer.apply_chat_template(history, add_generation_prompt=True,
                                                              enable_thinking=self.thinking_enabled)
        request_id = received[0]['id']
        sampling_params = SamplingParams(max_tokens=8192)
        results_generator = self.engine.generate(prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
                                                 sampling_params=sampling_params, request_id=request_id)

        full_response = ""
        is_tool_call = False
        TOOL_START_TOKEN = "<tool_call>"
        TOOL_END_TOKEN = "</tool_call>"
        is_thinking = False
        THINK_START_TOKEN = "<think>"
        THINK_END_TOKEN = "</think>"

        try:
            async for request_output in results_generator:
                generated_text = request_output.outputs[0].text
                new_text = generated_text[len(full_response):]
                full_response = generated_text
                self.history[-1]['content'] = full_response

                # Check for a tool call first
                if not is_tool_call and TOOL_START_TOKEN in new_text:
                    is_tool_call = True

                if is_tool_call:
                    if TOOL_END_TOKEN in new_text:
                        await self.engine.abort(request_id)
                        tool_content_str = full_response.split(TOOL_START_TOKEN)[1].split(TOOL_END_TOKEN)[0]
                        confirmation_msg = self._execute_tool_call(tool_content_str)

                        self.history[-1]['content'] = full_response # Save the raw tool call
                        self.history.append({'role': 'tool', 'content': confirmation_msg})

                        # Send confirmation text to client and TTS
                        self.streams['client'].send({'status': 'GENERATING', 'text': confirmation_msg, 'id': request_id})
                        if 'tts' in self.streams:
                            self.streams['tts'].send({'status': 'FINISHED', 'text': confirmation_msg, 'id': request_id})
                        else: # If no TTS, we must send the FINISHED signal ourselves
                            self.streams['client'].send({'status': 'FINISHED', 'text': '', 'id': request_id})

                        # Break the loop, the rest of the logic will handle audio forwarding
                        break
                    else:
                        # Still waiting for the end of the tool call, do nothing
                        continue

                # --- Regular text processing ---

                # Handle thinking state transitions and text filtering
                if self.thinking_enabled:
                    processed_text = ""
                    # Process chunk by chunk in case multiple tokens are in new_text
                    while new_text:
                        if not is_thinking:
                            if THINK_START_TOKEN in new_text:
                                before, after = new_text.split(THINK_START_TOKEN, 1)
                                processed_text += before

                                # Handle the "thinking" audio
                                thinking_text = "Hmm... let me think..."
                                self.streams['client'].send({'status': 'GENERATING', 'text': thinking_text, 'id': request_id})
                                if 'tts' in self.streams:
                                    self.streams['tts'].send({'status': 'GENERATING', 'text': thinking_text, 'id': request_id})
                                    self.streams['tts'].send({'status': 'FINISHED', 'text': '', 'id': request_id})

                                    # Wait for and forward the "thinking" audio, suppressing the FINISHED status
                                    waiting_for_think_audio = True
                                    while waiting_for_think_audio:
                                        tts_msgs = self.streams['tts'].recv()
                                        for tts_msg in tts_msgs:
                                            if tts_msg.get('status') == 'FINISHED':
                                                final_chunk = tts_msg.copy()
                                                final_chunk['status'] = 'GENERATING'
                                                if final_chunk.get('audio'):
                                                    self.streams['client'].send(final_chunk)
                                                waiting_for_think_audio = False
                                            else:
                                                self.streams['client'].send(tts_msg)
                                        if waiting_for_think_audio:
                                            await asyncio.sleep(POLL_INTERVAL)

                                new_text = after
                                is_thinking = True
                            else:
                                processed_text += new_text
                                new_text = ""
                        else: # is_thinking
                            if THINK_END_TOKEN in new_text:
                                _, after = new_text.split(THINK_END_TOKEN, 1)
                                new_text = after
                                is_thinking = False
                            else: # Discard content inside <think>
                                new_text = ""
                    new_text = processed_text

                is_finished = request_output.finished
                if new_text or is_finished:
                    # Send the processed text to client and TTS
                    # The final FINISHED status is sent by the TTS server, not here.
                    self.streams['client'].send({'status': 'GENERATING', 'text': new_text, 'id': request_id})
                    if 'tts' in self.streams:
                        tts_status = 'FINISHED' if is_finished else 'GENERATING'
                        self.streams['tts'].send({'status': tts_status, 'text': new_text, 'id': request_id})

            # --- End of async for loop ---

            # After the loop, forward all remaining audio from TTS to the client
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
                        if self.streams['tts'].closed and self.streams['tts'].received_q.empty():
                            break

        except asyncio.CancelledError:
            await self.engine.abort(request_id)
            raise


if __name__ == '__main__':
    asyncio.run(ChatServer('0.0.0.0', '12346', TTS_URI).serve_forever())
