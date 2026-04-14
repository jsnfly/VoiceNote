import asyncio
import logging
import os
import re
import threading
from typing import List, Union

from llama_cpp import Llama, StoppingCriteriaList
from llama_cpp.llama_chat_format import Jinja2ChatFormatter, Qwen25VLChatHandler

from server.base_server import BaseServer
from server.chat.tools import TOOLS
from server.utils.streaming_connection import POLL_INTERVAL
from server.utils.message import Message

logger = logging.getLogger(__name__)

CHAT_MODEL = os.getenv('CHAT_MODEL', './models/chat/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf')
CHAT_MMPROJ = os.getenv('CHAT_MMPROJ')
CHAT_CONTEXT_LENGTH = int(os.getenv('CHAT_CONTEXT_LENGTH', '16384'))
CHAT_MAX_TOKENS = int(os.getenv('CHAT_MAX_TOKENS', '2048'))
CHAT_GPU_LAYERS = int(os.getenv('CHAT_GPU_LAYERS', '-1'))
CHAT_FLASH_ATTN = os.getenv('CHAT_FLASH_ATTN', '1') not in {'0', 'false', 'False'}
TTS_URI = 'ws://tts:12347'
BASE_SYSTEM_PROMPT = """You are a helpful, smart and funny assistant talking directly to the user by leveraging
speech-to-text and text-to-speech. So keep your responses concise like in a real conversation and do not use any
spechial characters (including dashes, asteriks and so on) or emojis as they can not be expressed by the
text-to-speech component. You only call tools (see below) if the user asks for it.""".replace("\n", " ")


def _strip_partial_control_tail(text: str) -> str:
    for tag in ('<think>', '</think>', '<tool_call>', '</tool_call>'):
        for prefix_len in range(1, len(tag)):
            if text.endswith(tag[:prefix_len]):
                return text[:-prefix_len]
    return text


def _extract_visible_text(text: str) -> str:
    visible = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    visible = re.sub(r'<tool_call>.*?</tool_call>', '', visible, flags=re.DOTALL)
    for tag in ('<think>', '<tool_call>'):
        tag_idx = visible.rfind(tag)
        if tag_idx != -1:
            visible = visible[:tag_idx]
    return _strip_partial_control_tail(visible)


def _coerce_tool_argument(name: str, value: str):
    value = value.strip()
    if name == 'thinking_enabled':
        lowered = value.lower()
        if lowered == 'true':
            return True
        if lowered == 'false':
            return False
    return value


def _parse_tool_call(tool_call_str: str) -> dict:
    match = re.fullmatch(r'<function=([^>]+)>\s*(.*?)\s*</function>', tool_call_str, re.DOTALL)
    if not match:
        raise ValueError('Tool call did not match the expected Qwen function format.')

    arguments = {}
    for parameter_name, parameter_value in re.findall(
        r'<parameter=([^>]+)>\s*(.*?)\s*</parameter>', match.group(2), re.DOTALL
    ):
        arguments[parameter_name.strip()] = _coerce_tool_argument(parameter_name, parameter_value)

    return {
        'name': match.group(1).strip(),
        'arguments': arguments,
    }


def _get_chat_formatter(llm: Llama) -> Jinja2ChatFormatter:
    template = llm.metadata.get('tokenizer.chat_template')
    if not template:
        raise ValueError('The GGUF model does not expose a tokenizer.chat_template.')

    bos_token = llm.detokenize([llm.token_bos()], special=True).decode('utf-8', errors='ignore')
    eos_token = llm.detokenize([llm.token_eos()], special=True).decode('utf-8', errors='ignore')
    return Jinja2ChatFormatter(template=template, bos_token=bos_token, eos_token=eos_token,
                               stop_token_ids=[llm.token_eos()])


class ChatServer(BaseServer):

    def __init__(self, host: str, port: int, tts_uri: Union[str, None] = None):
        super().__init__("chat", host, port)
        self.custom_system_prompt = None
        self.thinking_enabled = False
        self.reset_conversation()

        chat_handler = None
        if CHAT_MMPROJ:
            chat_handler = Qwen25VLChatHandler(clip_model_path=CHAT_MMPROJ, verbose=False)

        self.llm = Llama(model_path=CHAT_MODEL, n_ctx=CHAT_CONTEXT_LENGTH, n_batch=512,
                         n_ubatch=512, n_gpu_layers=CHAT_GPU_LAYERS, offload_kqv=True,
                         chat_handler=chat_handler, flash_attn=CHAT_FLASH_ATTN, verbose=False)
        self.chat_formatter = _get_chat_formatter(self.llm)

        if tts_uri is not None:
            self.connections = {'tts': tts_uri}

    def _get_system_prompt(self) -> str:
        if self.custom_system_prompt:
            return f"{BASE_SYSTEM_PROMPT} {self.custom_system_prompt}"
        return BASE_SYSTEM_PROMPT

    def _reset_settings(self) -> None:
        self.custom_system_prompt = None
        self.thinking_enabled = False

    def reset_conversation(self) -> None:
        self.history = [{'role': 'system', 'content': self._get_system_prompt()}]

    def _create_completion_stream(self, cancel_event: threading.Event):
        formatter_response = self.chat_formatter(messages=self.history, tools=TOOLS,
                                                 enable_thinking=self.thinking_enabled)
        prompt_tokens = self.llm.tokenize(formatter_response.prompt.encode('utf-8'),
                                          add_bos=not formatter_response.added_special, special=True)
        stop = formatter_response.stop or []
        if isinstance(stop, str):
            stop = [stop]

        stopping_criteria = []
        if formatter_response.stopping_criteria is not None:
            stopping_criteria.extend(formatter_response.stopping_criteria)
        stopping_criteria.append(lambda *_args: cancel_event.is_set())

        return self.llm.create_completion(
            prompt=prompt_tokens,
            max_tokens=CHAT_MAX_TOKENS,
            temperature=1.0 if self.thinking_enabled else 0.7,
            top_p=0.95 if self.thinking_enabled else 0.8,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            repeat_penalty=1.0,
            stop=stop,
            stream=True,
            stopping_criteria=StoppingCriteriaList(stopping_criteria),
        )

    async def _stream_completion_chunks(self):
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        cancel_event = threading.Event()

        def run_generation() -> None:
            try:
                for chunk in self._create_completion_stream(cancel_event):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
                    if cancel_event.is_set():
                        break
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        generation_task = asyncio.create_task(asyncio.to_thread(run_generation))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        except asyncio.CancelledError:
            cancel_event.set()
            try:
                await generation_task
            except Exception:
                pass
            raise
        else:
            await generation_task

    def _apply_tool_call(self, tool_call: dict) -> str:
        if tool_call['name'] != 'update_chat_settings':
            return ''

        arguments = tool_call['arguments']
        updates = []

        system_prompt = arguments.get('system_prompt')
        if system_prompt is not None and system_prompt != self.custom_system_prompt:
            self.custom_system_prompt = system_prompt
            self.reset_conversation()
            updates.append('updated how I should respond going forward')

        thinking_enabled = arguments.get('thinking_enabled')
        if thinking_enabled is not None and thinking_enabled != self.thinking_enabled:
            self.thinking_enabled = thinking_enabled
            updates.append(f"thinking mode {'enabled' if thinking_enabled else 'disabled'}")

        return '. '.join(updates).capitalize() + '.' if updates else ''

    def _recv_client_messages(self) -> List[Message.DataDict]:
        text_messages = []
        for msg in super()._recv_client_messages():
            if msg.get('action') == 'NEW CONVERSATION':
                self._reset_settings()
                self.reset_conversation()
            else:
                text_messages.append(msg)
        return text_messages

    def _get_cutoff_idx(self, received: List[Message.DataDict]) -> int:
        return int(len(received) > 0)

    async def _run_workload(self, received: List[Message.DataDict]) -> None:
        self.history.append({'role': 'user', 'content': received[0]['text']})
        print(self.history)
        request_id = received[0]['id']
        full_text = ''
        visible_text = ''
        tool_calls = dict()
        is_finished = False
        thought_announced = False
        try:
            async for chunk in self._stream_completion_chunks():
                stream_text = ''
                choice = chunk['choices'][0]
                new_text = choice['text']
                full_text += new_text

                if self.thinking_enabled and not thought_announced and '<think>' in full_text:
                    thought_announced = True
                    stream_text = 'Let me think about that.'

                for tool_call_str in re.findall(r'<tool_call>(.+?)</tool_call>', full_text, re.DOTALL):
                    tool_call_str = tool_call_str.strip()
                    if tool_call_str in tool_calls:
                        continue

                    try:
                        tool_call = _parse_tool_call(tool_call_str)
                    except ValueError as exc:
                        tagged_tool_call = f'<tool_call>{tool_call_str}</tool_call>'
                        match_idx = full_text.find(tagged_tool_call)
                        context_start = max(0, match_idx - 200) if match_idx != -1 else max(0, len(full_text) - 400)
                        context_end = min(len(full_text), match_idx + len(tagged_tool_call) + 200) if match_idx != -1 else len(full_text)
                        logger.exception(
                            'Failed to parse tool call: %s\nraw_tool_call=%r\nfull_text_context=%r',
                            exc,
                            tool_call_str,
                            full_text[context_start:context_end],
                        )
                        raise

                    tool_response = self._apply_tool_call(tool_call)
                    tool_calls[tool_call_str] = {
                        'name': tool_call['name'],
                        'content': tool_response,
                    }
                    stream_text += tool_response

                next_visible_text = _extract_visible_text(full_text)
                visible_delta = next_visible_text[len(visible_text):]
                visible_text = next_visible_text
                if visible_delta:
                    if self.history[-1]['role'] == 'assistant':
                        self.history[-1]['content'] += visible_delta
                    else:
                        self.history.append({'role': 'assistant', 'content': visible_delta.lstrip("\n")})
                    stream_text += visible_delta

                is_finished = choice['finish_reason'] is not None

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
                    if 'tts' in self.streams:
                        self.streams['tts'].send({
                            'status': 'FINISHED' if is_finished else 'GENERATING',
                            'text': stream_text,
                            'id': request_id
                    })
                await asyncio.sleep(POLL_INTERVAL)

            for tool_call in tool_calls.values():
                if tool_call.get('content', ''):
                    self.history.append({'role': 'tool', **tool_call})

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
            raise

if __name__ == '__main__':
    asyncio.run(ChatServer('0.0.0.0', '12346', TTS_URI).serve_forever())
