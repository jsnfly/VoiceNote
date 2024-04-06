import asyncio
import websockets
from concurrent.futures import ThreadPoolExecutor
from websockets.server import serve, WebSocketServerProtocol
from typing import Any, Callable, List

from server.utils.streaming_connection import POLL_INTERVAL, StreamingConnection, StreamReset
from server.utils.message import Message


class BaseServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.connections = {}  # Connection URIs to other servers.

    async def serve_forever(self) -> None:
        async with serve(self.handle_connection, self.host, self.port):
            await asyncio.Future()

    async def handle_connection(self, client_connection: WebSocketServerProtocol) -> None:
        print(f"Connection from {client_connection.remote_address}")
        self.streams = {}
        for key, uri in self.connections.items():
            self.streams[key] = await self.setup_connection(uri)
        self.streams['client'] = StreamingConnection(client_connection)

        _, pending = await asyncio.wait(self._create_tasks(), return_when=asyncio.FIRST_COMPLETED)
        StreamingConnection.cancel_tasks(pending)
        self.streams = {}

    @staticmethod
    async def setup_connection(uri: str) -> StreamingConnection:
        while True:
            try:
                connection = await websockets.connect(uri)
                break
            except ConnectionRefusedError:
                await asyncio.sleep(POLL_INTERVAL)
        return StreamingConnection(connection)

    def _create_tasks(self) -> List[asyncio.Task]:
        streaming_tasks = [asyncio.create_task(stream.run(), name=key) for key, stream in self.streams.items()]
        return streaming_tasks + [asyncio.create_task(self._handle_workload())]

    @staticmethod
    async def run_blocking_function_in_thread(blocking_fn: Callable, fn_args: List[Any] = []) -> Any:
        with ThreadPoolExecutor() as pool:
            # Could also use `None` as executor instead of `pool` which would also use a ThreadPoolExecutor by
            # default. However, this seems more explicit.
            result = await asyncio.get_running_loop().run_in_executor(pool, blocking_fn, *fn_args)
        return result

    async def _handle_workload(self) -> None:
        received = []
        while True:
            try:
                received += self._recv_client_messages()
                cutoff = self._get_cutoff_idx(received)
                if cutoff > 0:
                    await self._run_workload(received[:cutoff])
                    received = received[cutoff:]
                else:
                    await asyncio.sleep(POLL_INTERVAL)
            except StreamReset:
                received = []
                [stream.reset() for key, stream in self.streams.items() if key != 'client']
            except ConnectionError:
                break

    def _recv_client_messages(self) -> List[Message.DataDict]:
        return self.streams['client'].recv()

    def _get_cutoff_idx(self, received: List[Message.DataDict]) -> int:
        """ Return the index of the last element that should be included in the workload.
        """
        return len(received)

    def _run_workload(self, received: List[Message.DataDict]) -> None:
        raise NotImplementedError
