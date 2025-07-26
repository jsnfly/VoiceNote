import asyncio
import logging
from datetime import datetime
from hashlib import md5
from typing import List, Union
from websockets.server import WebSocketServerProtocol
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from server.utils import message
from server.utils.misc import BASE_DIR

LOG_DIR = BASE_DIR / 'logs'
RESET_SENTINEL = object()


class StreamReset(Exception):
    def __init__(self, message: str, id_: str):
        super().__init__(message)
        self.id = id_


class StreamingConnection:
    def __init__(self, name: str, connection: Union[WebSocketClientProtocol, WebSocketServerProtocol]):
        self._setup_logger(name)
        self.connection = connection
        self.received_q = asyncio.Queue()
        self.ready_to_send_q = asyncio.Queue()
        self.closed = False
        self.communication_id = None
        self._tasks: List[asyncio.Task] = []

    async def run(self) -> None:
        self._tasks = [
            asyncio.create_task(self._recv_loop()),
            asyncio.create_task(self._send_loop())
        ]
        try:
            done, pending = await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                # Propagate exception if any
                if task.exception():
                    raise task.exception()
            for task in pending:
                task.cancel()
        except ConnectionClosedOK:
            self.logger.info("Connection closed cleanly.")
        except ConnectionClosedError as e:
            self.logger.error(f"Connection closed with error: {e}")
        finally:
            self.closed = True
            for task in self._tasks:
                if not task.done():
                    task.cancel()

    async def _recv_loop(self) -> None:
        async for raw_msg in self.connection:
            data = message.decode(raw_msg)
            encoded_msg = message.encode(data)
            self.logger.debug(f"Received message {md5(encoded_msg.encode()).hexdigest()} with status {data.get('status')}")

            if data.get('status') == 'RESET':
                self.reset(data['id'], propagate=False)
                await self.received_q.put(RESET_SENTINEL)
            elif self._is_valid_msg(data.get('id')):
                await self.received_q.put(data)
            else:
                self.logger.warning(f"Discarding msg with id {data.get('id')}.")

    async def _send_loop(self) -> None:
        while True:
            data = await self.ready_to_send_q.get()
            encoded_msg = message.encode(data)
            await self.connection.send(encoded_msg)
            self.logger.debug(f"Sent message {md5(encoded_msg.encode()).hexdigest()} with status {data.get('status')}")
            self.ready_to_send_q.task_done()

    def send(self, data: message.DataDict) -> None:
        if self.closed:
            raise ConnectionError("Connection is closed.")

        if not self._is_valid_msg(data.get('id')):
            raise StreamReset("Invalid message ID", self.communication_id)

        self.ready_to_send_q.put_nowait(data)

    def recv(self) -> List[message.DataDict]:
        if self.closed and self.received_q.empty():
            raise ConnectionError("Connection is closed.")

        received = []
        while not self.received_q.empty():
            try:
                received.append(self.received_q.get_nowait())
            except asyncio.QueueEmpty:
                break
        return received

    async def close(self) -> None:
        await self.connection.close()

    def reset(self, id_: str, propagate: bool = True) -> None:
        self.communication_id = id_
        # Clear queues
        while not self.received_q.empty():
            try:
                self.received_q.get_nowait()
            except asyncio.QueueEmpty:
                break
        while not self.ready_to_send_q.empty():
            try:
                self.ready_to_send_q.get_nowait()
            except asyncio.QueueEmpty:
                break

        if propagate:
            self.ready_to_send_q.put_nowait({'id': id_, 'status': 'RESET'})

    def _is_valid_msg(self, id_: str) -> bool:
        return (self.communication_id is None) or (id_ == self.communication_id)

    def _setup_logger(self, name):
        LOG_DIR.mkdir(exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        # Avoid adding handlers multiple times if logger already exists
        if not self.logger.handlers:
            file_handler = logging.FileHandler(f"{LOG_DIR}/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            formatter = logging.Formatter('%(asctime)s,%(msecs)03d - %(levelname)s - %(message)s',
                                          datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
