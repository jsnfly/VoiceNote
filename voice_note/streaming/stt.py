import asyncio
import time
import websockets

from base_server import BaseServer, POLL_INTERVAL
from typing import List, Union
from utils.message import Message


class TTSServer(BaseServer):
    def __init__(self, host: str, port: int, chat_uri: Union[str, None] = None):
        super().__init__(host, port)
        self.received = []
        self.chat_uri = chat_uri

    async def handle_workload(self) -> Message.DataDict:
        while True:
            end_idx = self._get_end_idx()
            if end_idx == -1:
                self.received += self.recv_from_client()
                await asyncio.sleep(POLL_INTERVAL)
            else:
                messages = self.received[:end_idx + 1]
                self.received = self.received[end_idx + 1:]
                transcription = await self.run_blocking_function_in_thread(self.dummy_transcribe, [messages])
                result = {"status": "FINISHED", "transcription": transcription}
                if self.chat_uri is not None:
                    async for msg in self._get_chat_response(result):
                        self.send_to_client(msg)
                else:
                    self.send_to_client(result)

    def _get_end_idx(self) -> int:
        return next((idx for idx, msg in enumerate(self.received) if msg["status"] == "FINISHED"), -1)

    def dummy_transcribe(self, messages: List[Message]) -> str:
        time.sleep(1)
        return b''.join([msg["audio"] for msg in messages]).decode()

    async def _get_chat_response(self, msg: Message.DataDict) -> None:
        async with websockets.connect(self.chat_uri) as chat_websocket:
            await chat_websocket.send(Message(msg).encode())
            while True:
                response = Message.decode(await chat_websocket.recv())
                yield response
                if response["status"] == "FINISHED":
                    break


if __name__ == '__main__':
    asyncio.run(TTSServer('0.0.0.0', '12345', 'ws://localhost:12346').serve_forever())
