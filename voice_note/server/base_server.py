import asyncio
import threading
import websockets
from websockets.server import serve, WebSocketServerProtocol
from typing import Any, Dict

from server.utils.streaming_connection import StreamingConnection


POLL_INTERVAL = 0.005  # Seconds


class ThreadExecutor:
    def __init__(self):
        self.cancel_event = threading.Event()

    async def run(self, *fn_args) -> Any:
        self.cancel_event.clear()
        try:
            result = await asyncio.to_thread(self.blocking_fn, *fn_args)
            return result
        except asyncio.CancelledError:
            self.cancel_event.set()
            raise

    def blocking_fn(self, *args, **kwargs) -> Any:
        raise NotImplementedError


class BaseServer:
    def __init__(self, name: str, host: str, port: int):
        self.name = name
        self.host = host
        self.port = port
        self.connections: Dict[str, str] = {}  # Connection URIs to other servers.
        self.streams: Dict[str, StreamingConnection] = {}

    async def serve_forever(self) -> None:
        async with serve(self.handle_connection, self.host, self.port):
            print(f"{self.name} server started at {self.host}:{self.port}")
            await asyncio.Future()  # Run forever

    async def handle_connection(self, client_connection: WebSocketServerProtocol) -> None:
        print(f"Connection from {client_connection.remote_address}")

        # Close any existing streams before creating new ones
        for stream in self.streams.values():
            await stream.close()
        self.streams.clear()

        try:
            # Setup connections to other servers
            for key, uri in self.connections.items():
                self.streams[key] = await self.setup_connection(key, uri)

            # Setup client connection
            self.streams['client'] = StreamingConnection(f"{self.name}_client", client_connection)

            # Start connection background tasks and the main loop
            tasks = [asyncio.create_task(stream.run(), name=key) for key, stream in self.streams.items()]
            main_task = asyncio.create_task(self._main_loop(), name=f"{self.name}_main_loop")
            tasks.append(main_task)

            # Wait for any task to complete (which signals the end of the session)
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            # Propagate exceptions
            for task in done:
                if task.exception():
                    raise task.exception()

        except Exception as e:
            print(f"An error occurred in handle_connection: {e}")
        finally:
            print("Closing connections...")
            # Cancel all pending tasks to ensure clean shutdown
            for task in pending:
                task.cancel()
            # Ensure all streams are closed
            for stream in self.streams.values():
                await stream.close()
            self.streams.clear()
            print("Connection handling finished.")

    async def setup_connection(self, connection_name: str, uri: str) -> StreamingConnection:
        while True:
            try:
                connection = await websockets.connect(uri)
                print(f"Successfully connected to {connection_name} at {uri}")
                return StreamingConnection(f"{self.name}_{connection_name}", connection)
            except ConnectionRefusedError:
                print(f"Connection to {connection_name} refused. Retrying in {POLL_INTERVAL * 100}s...")
                await asyncio.sleep(POLL_INTERVAL * 100)

    async def _main_loop(self) -> None:
        """
        Main logic loop to be implemented by subclasses.
        This loop will typically receive messages from the 'client' stream
        and orchestrate the work.
        """
        raise NotImplementedError
