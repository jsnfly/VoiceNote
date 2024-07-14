import asyncio

from websockets.server import WebSocketServerProtocol
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from queue import SimpleQueue, Empty
from typing import List, Union
from server.utils.message import Message

POLL_INTERVAL = 0.005  # Seconds


class StreamReset(Exception):
    def __init__(self, message, id_):
        super().__init__(message)
        self.id = id_


class StreamingConnection:
    def __init__(self, connection: Union['StreamingConnection', WebSocketClientProtocol, WebSocketServerProtocol]):
        self.connection = connection.connection if isinstance(connection, StreamingConnection) else connection
        self.received_q = SimpleQueue()
        self.ready_to_send_q = SimpleQueue()
        self.closed = False
        self.communication_id = None

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

        if msg.data.get('status') == 'RESET':
            self.reset(msg['id'], propagate=False)
        elif self._is_valid_msg(msg.data.get('id')):
            self.received_q.put(msg.data)
        else:
            print(f"Discarding msg with id {msg.data.get('id')}.")

    async def _send_from_queue(self) -> None:
        try:
            msg = Message(self.ready_to_send_q.get_nowait())
            await self.connection.send(msg.encode())
        except Empty:
            await asyncio.sleep(POLL_INTERVAL)

    def send(self, data: Message.DataDict) -> None:
        if self.closed:
            raise ConnectionError
        elif self._is_valid_msg(data.get('id')):
            self.ready_to_send_q.put(data)
        else:
            raise StreamReset("Invalid message ID", self.communication_id)

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

    def reset(self, id_, propagate=True):
        self.communication_id = id_
        self.received_q = SimpleQueue()
        self.ready_to_send_q = SimpleQueue()
        if propagate:
            self.send({'id': id_, 'status': 'RESET'})

    def _is_valid_msg(self, id_):
        return (self.communication_id is None) or (id_ == self.communication_id)
