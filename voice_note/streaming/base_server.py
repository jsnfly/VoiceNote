import asyncio
from concurrent.futures import ThreadPoolExecutor
from websockets.server import serve, WebSocketServerProtocol
from typing import Any, Callable, List

from utils.streaming_connection import StreamingConnection


class BaseServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    async def serve_forever(self) -> None:
        async with serve(self.handle_connection, self.host, self.port):
            await asyncio.Future()

    async def handle_connection(self, connection: WebSocketServerProtocol) -> None:
        print(f"Connection from {connection.remote_address}")
        connection = StreamingConnection(connection)
        await asyncio.gather(connection.run(), self._handle_workload(connection))

    @staticmethod
    async def run_blocking_function_in_thread(blocking_fn: Callable, fn_args: List[Any] = []) -> Any:
        with ThreadPoolExecutor() as pool:
            # Could also use `None` as executor instead of `pool` which would also use a ThreadPoolExecutor by
            # default. However, this seems more explicit.
            result = await asyncio.get_running_loop().run_in_executor(pool, blocking_fn, *fn_args)
        return result

    async def _handle_workload(self, connection: StreamingConnection) -> None:
        raise NotImplementedError
