import asyncio
from websockets.server import serve, WebSocketServerProtocol
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from queue import SimpleQueue, Empty
from typing import List
from utils.message import Message

POLL_INTERVAL = 0.01  # Seconds


class StreamingConnection:
    def __init__(self, connection: WebSocketServerProtocol):
        self.connection = connection
        self.received = SimpleQueue()
        self.ready_to_send = SimpleQueue()

    async def recv_to_queue(self) -> None:
        msg = Message.from_data_string(await self.connection.recv())
        self.received.put(msg.data)

    async def send_from_queue(self) -> None:
        try:
            await self.connection.send(Message(self.ready_to_send.get_nowait()).encode())
        except Empty:
            await asyncio.sleep(POLL_INTERVAL)

    async def run_communication(self) -> None:
        while True:
            try:
                tasks = [asyncio.create_task(self.recv_to_queue()), asyncio.create_task(self.send_from_queue())]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                self._cancel_tasks(pending)
                for task in done:
                    if task.exception() is not None:
                        raise task.exception()
            except ConnectionClosedOK:
                break
            except ConnectionClosedError:
                break
            finally:
                self._cancel_tasks(tasks)

    @staticmethod
    def _cancel_tasks(tasks: List[asyncio.Task]) -> None:
        for task in tasks:
            if not task.done():
                task.cancel()


class BaseServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    async def serve_forever(self) -> None:
        async with serve(self.handle_connection, self.host, self.port):
            await asyncio.Future()

    async def handle_connection(self, connection: WebSocketServerProtocol) -> None:
        print(f"Connection from {connection.remote_address}")
        self._client_connection = StreamingConnection(connection)
        await asyncio.gather(self._client_connection.run_communication(), self.handle_workload())

    async def handle_workload(self) -> None:
        raise NotImplementedError

    def send_to_client(self, data: Message.DataDict) -> None:
        self._client_connection.ready_to_send.put(data)

    def recv_from_client(self) -> List[Message.DataDict]:
        received = []
        while True:
            try:
                received.append(self._client_connection.received.get_nowait())
            except Empty:
                break
        return received


class SimplexServer(BaseServer):
    """ Communication only in one direction at a time.
    """

    def __init__(self, host: str, port: int):
        super().__init__(host, port)
        self.received = []

    async def handle_workload(self) -> None:
        while True:
            self.received += self.recv_from_client()
            if self.received and self.received[-1].get("status") == "FINISHED":
                # TODO: handle more data has already been sent.
                await self.process()
                self.received = []
            else:
                await asyncio.sleep(POLL_INTERVAL)

            # TODO: do this and the loop in base class?
            if not self._client_connection.connection.open:
                break

    async def process(self) -> None:
        raise NotImplementedError
