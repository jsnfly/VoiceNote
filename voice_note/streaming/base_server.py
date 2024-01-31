import asyncio
import websockets
from concurrent.futures import ThreadPoolExecutor
from websockets.server import serve, WebSocketServerProtocol
from typing import Any, Callable, List

from utils.streaming_connection import POLL_INTERVAL, StreamingConnection


class BaseServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.connections = {}  # Connections to other servers.

    async def serve_forever(self) -> None:
        async with serve(self.handle_connection, self.host, self.port):
            await asyncio.Future()

    async def handle_connection(self, client_connection: WebSocketServerProtocol) -> None:
        print(f"Connection from {client_connection.remote_address}")
        self.connections['client'] = client_connection
        await self._setup_connections()
        await asyncio.gather(*[conn.run() for conn in self.connections.values()], self._handle_workload())

    async def _setup_connections(self) -> None:
        for key, connection in self.connections.items():
            if isinstance(connection, str):
                while True:
                    try:
                        connection = await websockets.connect(connection)
                        break
                    except ConnectionRefusedError:
                        await asyncio.sleep(POLL_INTERVAL)

            self.connections[key] = StreamingConnection(connection)

    @staticmethod
    async def run_blocking_function_in_thread(blocking_fn: Callable, fn_args: List[Any] = []) -> Any:
        with ThreadPoolExecutor() as pool:
            # Could also use `None` as executor instead of `pool` which would also use a ThreadPoolExecutor by
            # default. However, this seems more explicit.
            result = await asyncio.get_running_loop().run_in_executor(pool, blocking_fn, *fn_args)
        return result

    async def _handle_workload(self) -> None:
        raise NotImplementedError
