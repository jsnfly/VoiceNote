import asyncio
import time
import websockets

from websockets.server import WebSocketServerProtocol
from typing import List, Union, Callable

from base_server import BaseServer, POLL_INTERVAL, StreamingConnectionHandler
from utils.message import Message


class STTHandler(StreamingConnectionHandler):
    def __init__(self, connection: WebSocketServerProtocol, get_chat_response: Union[Callable, None]):
        super().__init__(connection)

        self.received = []
        self.get_chat_rsponse = get_chat_response

    async def _handle_workload(self) -> Message.DataDict:
        while True:
            end_idx = self._get_end_idx()
            if end_idx == -1:
                self.received += self.recv_from_client()
                await asyncio.sleep(POLL_INTERVAL)
            else:
                messages = self.received[:end_idx + 1]
                self.received = self.received[end_idx + 1:]
                transcription = await BaseServer.run_blocking_function_in_thread(self.dummy_transcribe, [messages])
                result = {"status": "FINISHED", "transcription": transcription}
                if self.get_chat_rsponse is not None:
                    async for msg in self.get_chat_rsponse(result):
                        self.send_to_client(msg)
                else:
                    self.send_to_client(result)

    def _get_end_idx(self) -> int:
        return next((idx for idx, msg in enumerate(self.received) if msg["status"] == "FINISHED"), -1)

    def dummy_transcribe(self, messages: List[Message]) -> str:
        time.sleep(1)
        return b''.join([msg["audio"] for msg in messages]).decode()


class STTServer(BaseServer):
    def __init__(self, host: str, port: int, chat_uri: Union[str, None] = None):
        super().__init__(host, port, STTHandler)
        self.chat_uri = chat_uri

    async def handle_connection(self, connection: WebSocketServerProtocol) -> None:
        print(f"Connection from {connection.remote_address}")
        await self.handler_cls(connection, self.get_chat_response).run()

    async def get_chat_response(self, msg: Message.DataDict) -> None:
        async with websockets.connect(self.chat_uri) as chat_websocket:
            await chat_websocket.send(Message(msg).encode())
            while True:
                response = Message.decode(await chat_websocket.recv())
                yield response
                if response["status"] == "FINISHED":
                    break


if __name__ == '__main__':
    asyncio.run(STTServer('0.0.0.0', '12345', 'ws://localhost:12346').serve_forever())
