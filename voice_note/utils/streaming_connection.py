import asyncio

from websockets.server import WebSocketServerProtocol
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from queue import SimpleQueue, Empty
from typing import List, Union
from utils.message import Message

POLL_INTERVAL = 0.05  # Seconds


class StreamingConnection:
    def __init__(self, connection: Union[WebSocketClientProtocol, WebSocketServerProtocol]):
        self.connection = connection
        self.received_q = SimpleQueue()
        self.ready_to_send_q = SimpleQueue()
        self.closed = False

    async def run(self) -> None:
        while True:
            try:
                tasks = self._create_communication_tasks()
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if task.exception() is not None:
                        raise task.exception()
            except ConnectionClosedOK:
                self.closed = True
                break
            except ConnectionClosedError:
                self.closed = True
                break
            finally:
                self.cancel_tasks(tasks)

    def _create_communication_tasks(self) -> List[asyncio.Task]:
        return [asyncio.create_task(self._recv_to_queue()), asyncio.create_task(self._send_from_queue())]

    @staticmethod
    def cancel_tasks(tasks: List[asyncio.Task]) -> None:
        for task in tasks:
            if not task.done():
                task.cancel()

    async def _recv_to_queue(self) -> None:
        msg = Message.from_data_string(await self.connection.recv())
        self.received_q.put(msg.data)

    async def _send_from_queue(self) -> None:
        try:
            await self.connection.send(Message(self.ready_to_send_q.get_nowait()).encode())
        except Empty:
            await asyncio.sleep(POLL_INTERVAL)

    def send(self, data: Message.DataDict) -> None:
        if self.closed:
            raise ConnectionError
        self.ready_to_send_q.put(data)

    def recv(self) -> List[Message.DataDict]:
        if self.closed:
            raise ConnectionError
        received = []
        while True:
            try:
                received.append(self.received_q.get_nowait())
            except Empty:
                break
        return received

    async def close(self) -> None:
        await self.connection.close()