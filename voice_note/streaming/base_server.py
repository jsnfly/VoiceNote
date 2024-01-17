import asyncio
from concurrent.futures import ThreadPoolExecutor
from websockets.server import serve, WebSocketServerProtocol
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from queue import SimpleQueue, Empty
from typing import Any, Callable, List
from utils.message import Message

POLL_INTERVAL = 0.01  # Seconds


# TODO: separate ConnectionHandler and Server.
class BaseServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

        self.client = None
        self.received_q = SimpleQueue()
        self.ready_to_send_q = SimpleQueue()

    async def serve_forever(self) -> None:
        async with serve(self.handle_connection, self.host, self.port):
            await asyncio.Future()

    async def handle_connection(self, connection: WebSocketServerProtocol) -> None:
        print(f"Connection from {connection.remote_address}")
        self.client = connection
        await asyncio.gather(self._handle_client_communication(), self.handle_workload())

    async def _handle_client_communication(self) -> None:
        while True:
            try:
                tasks = self._create_tasks()
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

    def _create_tasks(self) -> List[asyncio.Task]:
        return [asyncio.create_task(self._recv_to_queue()), asyncio.create_task(self._send_from_queue())]

    async def _recv_to_queue(self) -> None:
        msg = Message.from_data_string(await self.client.recv())
        self.received_q.put(msg.data)

    async def _send_from_queue(self) -> None:
        try:
            await self.client.send(Message(self.ready_to_send_q.get_nowait()).encode())
        except Empty:
            await asyncio.sleep(POLL_INTERVAL)

    async def handle_workload(self) -> None:
        raise NotImplementedError

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

    @staticmethod
    def _cancel_tasks(tasks: List[asyncio.Task]) -> None:
        for task in tasks:
            if not task.done():
                task.cancel()

    @staticmethod
    async def run_blocking_function_in_thread(blocking_fn: Callable, fn_args: List[Any] = []):
        with ThreadPoolExecutor() as pool:
            # Could also use `None` as executor instead of `pool` which would also use a ThreadPoolExecutor by
            # default. However, this seems more explicit.
            result = await asyncio.get_running_loop().run_in_executor(pool, blocking_fn, *fn_args)
        return result
