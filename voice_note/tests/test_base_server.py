import asyncio
import pytest
import websockets
from voice_note.server.base_server import BaseServer
from voice_note.server.utils.streaming_connection import StreamingConnection

PORT = 7757
DOWNSTREAM_PORT = 7758
TIMEOUT = 2.0

class EchoServer(BaseServer):
    """A simple server that echoes messages back to the client."""
    async def _main_loop(self) -> None:
        client_stream = self.streams['client']
        while True:
            msg = await client_stream.received_q.get()
            client_stream.send(msg)
            client_stream.received_q.task_done()

class DownstreamServer(BaseServer):
    """A dummy downstream server that just accepts a connection."""
    def __init__(self, host: str, port: int, connection_event: asyncio.Event):
        super().__init__("downstream", host, port)
        self.connection_event = connection_event

    async def handle_connection(self, websocket: websockets.WebSocketServerProtocol) -> None:
        self.connection_event.set()
        # Keep the connection open until the client closes it
        try:
            await websocket.wait_closed()
        except websockets.exceptions.ConnectionClosed:
            pass

class MainServerWithDownstream(BaseServer):
    """A server that connects to a downstream service."""
    def __init__(self, host: str, port: int, downstream_uri: str):
        super().__init__("main", host, port)
        self.connections['downstream'] = downstream_uri

    async def _main_loop(self) -> None:
        # Just keep the connection alive
        await asyncio.Future()

class CrashingServer(BaseServer):
    """A server whose main loop raises an exception immediately."""
    async def _main_loop(self) -> None:
        raise ValueError("This is a test exception.")


@pytest.mark.asyncio
async def test_server_lifecycle_and_echo():
    """Tests if the server starts, echoes a message, and shuts down cleanly."""
    server = EchoServer("echo", "localhost", PORT)
    server_task = asyncio.create_task(server.serve_forever())
    await asyncio.sleep(0.1)  # Give server time to start

    try:
        async with websockets.connect(f"ws://localhost:{PORT}") as websocket:
            test_msg = {"type": "echo_test", "payload": "hello"}

            # Use StreamingConnection on the client side for consistency
            client_conn = StreamingConnection("test_client", websocket)
            client_task = asyncio.create_task(client_conn.run())

            client_conn.send(test_msg)

            # Wait for the response
            response = None
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < TIMEOUT:
                received = client_conn.recv()
                if received:
                    response = received[0]
                    break
                await asyncio.sleep(0.01)

            assert response is not None
            assert response == test_msg

            await client_conn.close()
            await client_task

    finally:
        server_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await server_task

@pytest.mark.asyncio
async def test_downstream_connection():
    """Tests if the server correctly connects to a downstream service."""
    connection_event = asyncio.Event()

    # Start the dummy downstream server
    downstream_server = DownstreamServer("localhost", DOWNSTREAM_PORT, connection_event)
    downstream_task = asyncio.create_task(downstream_server.serve_forever())

    # Start the main server which should connect to the downstream one
    main_server = MainServerWithDownstream("localhost", PORT, f"ws://localhost:{DOWNSTREAM_PORT}")
    main_server_task = asyncio.create_task(main_server.serve_forever())

    await asyncio.sleep(0.1) # Give servers time to start

    try:
        # Connect a client to the main server to trigger its connection logic
        async with websockets.connect(f"ws://localhost:{PORT}"):
            # Wait for the downstream server to report a connection
            await asyncio.wait_for(connection_event.wait(), timeout=TIMEOUT)

        assert connection_event.is_set()

    finally:
        main_server_task.cancel()
        downstream_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await main_server_task
        with pytest.raises(asyncio.CancelledError):
            await downstream_task

@pytest.mark.asyncio
async def test_main_loop_exception():
    """Tests that the server handles exceptions in the main loop gracefully."""
    server = CrashingServer("crashing", "localhost", PORT)
    server_task = asyncio.create_task(server.serve_forever())
    await asyncio.sleep(0.1)

    try:
        # We expect the connection to be closed cleanly by the server after the exception
        with pytest.raises(websockets.exceptions.ConnectionClosedOK):
            async with websockets.connect(f"ws://localhost:{PORT}") as websocket:
                # The server's handle_connection should catch the exception from _main_loop
                # and close the connection. We wait to see if it does.
                await websocket.recv()

    finally:
        server_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await server_task
