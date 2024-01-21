import asyncio
from concurrent.futures import ThreadPoolExecutor
from websockets.server import serve, WebSocketServerProtocol
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from queue import SimpleQueue, Empty
from typing import Any, Callable, List
from utils.message import Message

POLL_INTERVAL = 0.01  # Seconds


class StreamingConnectionHandler:
    def __init__(self, connection: WebSocketServerProtocol):
        self.connection = connection
        self.received_q = SimpleQueue()
        self.ready_to_send_q = SimpleQueue()

    async def run(self):
        await asyncio.gather(self._handle_client_communication(), self._handle_workload())

    async def _handle_client_communication(self) -> None:
        while True:
            try:
                tasks = self._create_communication_tasks()
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if task.exception() is not None:
                        raise task.exception()
            except ConnectionClosedOK:
                break
            except ConnectionClosedError:
                break
            finally:
                self._cancel_tasks(tasks)

    async def _handle_workload(self) -> None:
        raise NotImplementedError

    def _create_communication_tasks(self) -> List[asyncio.Task]:
        return [asyncio.create_task(self._recv_to_queue()), asyncio.create_task(self._send_from_queue())]

    @staticmethod
    def _cancel_tasks(tasks: List[asyncio.Task]) -> None:
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

    def send_to_client(self, data: Message.DataDict) -> None:
        self.ready_to_send_q.put(data)

    def recv_from_client(self) -> List[Message.DataDict]:
        received = []
        while True:
            try:
                received.append(self.received_q.get_nowait())
            except Empty:
                break
        return received


class BaseServer:
    def __init__(self, host: str, port: int, handler_cls: StreamingConnectionHandler):
        self.host = host
        self.port = port
        self.handler_cls = handler_cls

    async def serve_forever(self) -> None:
        async with serve(self.handle_connection, self.host, self.port):
            await asyncio.Future()

    async def handle_connection(self, connection: WebSocketServerProtocol) -> None:
        print(f"Connection from {connection.remote_address}")
        await self.handler_cls(connection).run()

    @staticmethod
    async def run_blocking_function_in_thread(blocking_fn: Callable, fn_args: List[Any] = []):
        with ThreadPoolExecutor() as pool:
            # Could also use `None` as executor instead of `pool` which would also use a ThreadPoolExecutor by
            # default. However, this seems more explicit.
            result = await asyncio.get_running_loop().run_in_executor(pool, blocking_fn, *fn_args)
        return result
