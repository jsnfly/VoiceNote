import asyncio
import json
import logging
import os
import shlex
import shutil
from collections import deque
from pathlib import Path
from typing import AsyncIterator, List, Union
from uuid import uuid4

from server.base_server import BaseServer
from server.utils.misc import BASE_DIR
from server.utils.message import Message
from server.utils.streaming_connection import POLL_INTERVAL


logger = logging.getLogger(__name__)

TTS_URI = os.getenv('TTS_URI', 'ws://localhost:12347')
PI_MODEL = os.getenv('PI_MODEL', 'Qwen3.5-9B-Q8_0')
CHAT_AGENT_TOOLS = os.getenv('CHAT_AGENT_TOOLS', 'read-only')
CHAT_AGENT_CWD = os.getenv('CHAT_AGENT_CWD', os.getcwd())
PI_AGENT_DIR = Path(os.getenv('PI_CODING_AGENT_DIR', BASE_DIR / 'pi-agent'))
LLAMACPP_BASE_URL = os.getenv('LLAMACPP_BASE_URL', 'http://localhost:8080/v1')
SERVER_DIR = BASE_DIR / 'server'
LOCAL_PI_COMMAND = SERVER_DIR / 'node_modules' / '.bin' / 'pi'

READ_ONLY_TOOLS = 'read,grep,find,ls'
CODING_TOOLS = 'read,write,edit,bash,grep,find,ls'


def _write_pi_models_config() -> None:
    PI_AGENT_DIR.mkdir(parents=True, exist_ok=True)
    models_path = PI_AGENT_DIR / 'models.json'
    models_path.write_text(json.dumps({
        'providers': {
            'llamacpp': {
                'baseUrl': LLAMACPP_BASE_URL,
                'api': 'openai-completions',
                'apiKey': 'llamacpp',
                'compat': {
                    'supportsDeveloperRole': False,
                    'supportsReasoningEffort': False,
                    'supportsUsageInStreaming': False,
                    'maxTokensField': 'max_tokens',
                    'thinkingFormat': 'qwen-chat-template',
                },
                'models': [{
                    'id': PI_MODEL,
                    'name': f'{PI_MODEL} llama.cpp',
                    'reasoning': True,
                    'input': ['text'],
                    'contextWindow': 16384,
                    'maxTokens': 2048,
                    'cost': {
                        'input': 0,
                        'output': 0,
                        'cacheRead': 0,
                        'cacheWrite': 0,
                    },
                }],
            },
        },
    }, indent=2) + '\n')


def _get_pi_command() -> List[str]:
    pi_command = os.getenv('PI_COMMAND')
    if pi_command:
        command = shlex.split(pi_command)
    elif LOCAL_PI_COMMAND.exists():
        command = [str(LOCAL_PI_COMMAND)]
    elif shutil.which('pi'):
        command = ['pi']
    elif shutil.which('npx'):
        command = ['npx', '-y', '@earendil-works/pi-coding-agent']
    else:
        command = ['pi']

    command += ['--mode', 'rpc', '--provider', 'llamacpp', '--model', PI_MODEL, '--thinking', 'off']

    if CHAT_AGENT_TOOLS == 'coding':
        command += ['--tools', CODING_TOOLS]
    elif CHAT_AGENT_TOOLS == 'read-only':
        command += ['--tools', READ_ONLY_TOOLS]
    elif CHAT_AGENT_TOOLS == 'none':
        command += ['--no-tools']
    elif CHAT_AGENT_TOOLS:
        command += ['--tools', CHAT_AGENT_TOOLS]

    return command


class PiRpcClient:
    def __init__(self, command: List[str], cwd: str):
        self.command = command
        self.cwd = cwd
        self.process: asyncio.subprocess.Process | None = None
        self.stderr_task: asyncio.Task | None = None
        self.stderr_lines: deque[str] = deque(maxlen=50)
        self.lock = asyncio.Lock()

    async def start(self) -> None:
        if self.process is not None and self.process.returncode is None:
            return

        logger.info('Starting Pi RPC process: %s', shlex.join(self.command))
        env = os.environ.copy()
        env.setdefault('PI_CODING_AGENT_DIR', str(PI_AGENT_DIR))
        env.setdefault('PI_SKIP_VERSION_CHECK', '1')
        env.setdefault('PI_TELEMETRY', '0')
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.command,
                cwd=self.cwd,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                'Could not start Pi. Install @earendil-works/pi-coding-agent (requires Node >= 22), '
                'install npx, or set PI_COMMAND to the Pi executable.'
            ) from exc
        self.stderr_task = asyncio.create_task(self._collect_stderr())

    async def close(self) -> None:
        if self.process is None:
            return

        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()

        if self.stderr_task is not None:
            self.stderr_task.cancel()
            try:
                await self.stderr_task
            except asyncio.CancelledError:
                pass

        self.process = None
        self.stderr_task = None

    async def _collect_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while True:
            line = await self.process.stderr.readline()
            if not line:
                return
            decoded = line.decode('utf-8', errors='replace').rstrip()
            self.stderr_lines.append(decoded)
            logger.info('pi stderr: %s', decoded)

    async def new_session(self) -> None:
        await self._send_command_and_wait({'type': 'new_session'})

    async def abort(self) -> None:
        await self._send_command_and_wait({'type': 'abort'})

    async def prompt(self, message: str) -> AsyncIterator[dict]:
        async with self.lock:
            command_id = await self._send_command({'type': 'prompt', 'message': message})
            prompt_accepted = False
            while True:
                event = await self._read_event()

                if event.get('type') == 'response' and event.get('id') == command_id:
                    if not event.get('success'):
                        raise RuntimeError(event.get('error', 'Pi rejected the prompt.'))
                    prompt_accepted = True
                    continue

                await self._handle_extension_ui_request(event)
                yield event

                if event.get('type') == 'agent_end':
                    return

                if event.get('type') == 'message_update':
                    assistant_event = event.get('assistantMessageEvent', {})
                    if assistant_event.get('type') == 'error':
                        reason = assistant_event.get('reason', 'error')
                        if not prompt_accepted:
                            raise RuntimeError(f'Pi prompt failed before acceptance: {reason}')

    async def _send_command_and_wait(self, command: dict) -> dict:
        async with self.lock:
            command_id = await self._send_command(command)
            while True:
                event = await self._read_event()
                await self._handle_extension_ui_request(event)
                if event.get('type') == 'response' and event.get('id') == command_id:
                    if not event.get('success'):
                        raise RuntimeError(event.get('error', f"Pi command failed: {command['type']}"))
                    return event

    async def _send_command(self, command: dict) -> str:
        await self.start()
        assert self.process is not None and self.process.stdin is not None

        command_id = str(uuid4())
        command = {'id': command_id, **command}
        self.process.stdin.write((json.dumps(command) + '\n').encode('utf-8'))
        await self.process.stdin.drain()
        return command_id

    async def _read_event(self) -> dict:
        assert self.process is not None and self.process.stdout is not None

        line = await self.process.stdout.readline()
        if not line:
            return_code = await self.process.wait()
            stderr_tail = '\n'.join(self.stderr_lines) if self.stderr_lines else '(no stderr)'
            raise RuntimeError(
                f'Pi RPC process exited with code {return_code}. Stderr:\n{stderr_tail}'
            )

        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            decoded = line.decode('utf-8', errors='replace').rstrip()
            raise RuntimeError(f'Pi RPC emitted invalid JSON: {decoded}') from exc

    async def _handle_extension_ui_request(self, event: dict) -> None:
        if event.get('type') != 'extension_ui_request':
            return

        method = event.get('method')
        if method not in {'select', 'confirm', 'input', 'editor'}:
            return

        assert self.process is not None and self.process.stdin is not None
        response = {'type': 'extension_ui_response', 'id': event['id'], 'cancelled': True}
        self.process.stdin.write((json.dumps(response) + '\n').encode('utf-8'))
        await self.process.stdin.drain()


class ChatServer(BaseServer):

    def __init__(self, host: str, port: int, tts_uri: Union[str, None] = None):
        super().__init__('chat', host, port)
        _write_pi_models_config()
        self.pi = PiRpcClient(_get_pi_command(), CHAT_AGENT_CWD)
        self.new_session_requested = False

        if tts_uri is not None:
            self.connections = {'tts': tts_uri}

    def _recv_client_messages(self) -> List[Message.DataDict]:
        text_messages = []
        for msg in super()._recv_client_messages():
            if msg.get('action') == 'NEW CONVERSATION':
                self.new_session_requested = True
            else:
                text_messages.append(msg)
        return text_messages

    def _get_cutoff_idx(self, received: List[Message.DataDict]) -> int:
        return int(len(received) > 0)

    async def _run_workload(self, received: List[Message.DataDict]) -> None:
        request_id = received[0]['id']
        user_text = received[0]['text']

        try:
            if self.new_session_requested:
                await self.pi.new_session()
                self.new_session_requested = False

            async for event in self.pi.prompt(user_text):
                stream_text = self._extract_text_delta(event)
                if stream_text:
                    self._send_text(request_id, stream_text, 'GENERATING')

                self._forward_tts_messages()
                await asyncio.sleep(POLL_INTERVAL)

            await self._finish_response(request_id)
        except asyncio.CancelledError:
            await self.pi.abort()
            raise
        except Exception:
            logger.exception('Pi chat workload failed.')
            self._send_text(request_id, 'Sorry, I ran into an error.', 'GENERATING')
            await self._finish_response(request_id)

    @staticmethod
    def _extract_text_delta(event: dict) -> str:
        if event.get('type') != 'message_update':
            return ''

        assistant_event = event.get('assistantMessageEvent', {})
        if assistant_event.get('type') != 'text_delta':
            return ''

        return assistant_event.get('delta', '')

    def _send_text(self, request_id: str, text: str, status: str) -> None:
        self.streams['client'].send({'status': 'GENERATING', 'text': text, 'id': request_id})
        if 'tts' in self.streams:
            self.streams['tts'].send({'status': status, 'text': text, 'id': request_id})

    async def _finish_response(self, request_id: str) -> None:
        if 'tts' not in self.streams:
            self.streams['client'].send({'status': 'FINISHED', 'text': '', 'id': request_id})
            return

        self.streams['tts'].send({'status': 'FINISHED', 'text': '', 'id': request_id})
        waiting_for_tts = True
        while waiting_for_tts:
            for msg in self.streams['tts'].recv():
                self.streams['client'].send(msg)
                if msg.get('status') == 'FINISHED':
                    waiting_for_tts = False
            await asyncio.sleep(POLL_INTERVAL)
            if self.streams['tts'].closed and self.streams['tts'].received_q.empty():
                break

    def _forward_tts_messages(self) -> None:
        if 'tts' not in self.streams:
            return

        for tts_msg in self.streams['tts'].recv():
            self.streams['client'].send(tts_msg)


if __name__ == '__main__':
    asyncio.run(ChatServer('0.0.0.0', '12346', TTS_URI).serve_forever())
