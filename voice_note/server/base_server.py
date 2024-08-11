import asyncio
import threading
import websockets
from concurrent.futures import ThreadPoolExecutor
from websockets.server import serve, WebSocketServerProtocol
from typing import Any, List

from server.utils.streaming_connection import POLL_INTERVAL, StreamingConnection, StreamReset
from server.utils.message import Message


class ThreadExecutor:
    def __init__(self):
        self.cancel_event = threading.Event()

    async def run(self, *fn_args) -> Any:
        self.cancel_event.clear()
        try:
            with ThreadPoolExecutor() as pool:
                # Could also use `None` as executor instead of `pool` which would also use a ThreadPoolExecutor by
                # default. However, this seems more explicit.
                result = await asyncio.get_running_loop().run_in_executor(pool, self.blocking_fn, *fn_args)
            return result
        except asyncio.CancelledError:
            self.cancel_event.set()

    def blocking_fn(self) -> Any:
        raise NotImplementedError


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

    async def _handle_workload(self) -> None:
        received = []
        while True:
            try:
                received += self._recv_client_messages()

                # Discard data for a previous id. Necessary, because the StreamReset (and with that the clearing of
                # `received`) happens only after it was attempted to send a response.
                if len(received) > 1 and received[0]['id'] != received[-1]['id']:
                    received = [data for data in received if data['id'] == received[-1]['id']]

                cutoff = self._get_cutoff_idx(received)
                if cutoff > 0:
                    # Reset other streams with new id.
                    [stream.reset(received[0]['id']) for key, stream in self.streams.items() if key != 'client']

                    workload = asyncio.create_task(self._run_workload(received[:cutoff]))
                    await workload
                    received = received[cutoff:]
                else:
                    await asyncio.sleep(POLL_INTERVAL)
            except StreamReset as e:
                # Only triggered when the current server tries to send something via a resetting connection. This could
                # also be just forwarding of messages from another server, which would trigger the reset command to be
                # sent to it.
                [stream.reset(e.id) for key, stream in self.streams.items() if key != 'client']
                received = []
                workload.cancel()
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
