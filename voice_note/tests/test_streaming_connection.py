import asyncio
import pytest
import websockets
from voice_note.server.utils.streaming_connection import StreamingConnection, StreamReset

PORT = 7756
TIMEOUT = 2.0  # Seconds.


async def receive_one(connection: StreamingConnection, timeout: float = TIMEOUT) -> dict:
    """Helper to receive one message from a connection."""
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        received = connection.recv()
        if received:
            assert len(received) == 1
            return received[0]
        await asyncio.sleep(0.01)
    raise asyncio.TimeoutError("Did not receive a message in time.")


@pytest.mark.asyncio
async def test_send_and_receive():
    """Tests basic message sending and receiving between client and server."""
    data = {"binary_text": "abcd".encode(), "text": "abcd"}
    msg = {"id": "1234", **data, "nested": data}
    client_done = asyncio.Event()

    async def client_handler(uri):
        async with websockets.connect(uri) as websocket:
            connection = StreamingConnection("client", websocket)
            run_task = asyncio.create_task(connection.run())

            connection.send(msg)
            received_msg = await receive_one(connection)
            assert received_msg == msg

            client_done.set()
            # Give server time to process client_done event before closing
            await asyncio.sleep(0.1)
            await connection.close()
            await run_task

    async def server_handler(websocket):
        connection = StreamingConnection("server", websocket)
        run_task = asyncio.create_task(connection.run())

        received_msg = await receive_one(connection)
        assert received_msg == msg
        connection.send(msg)

        await client_done.wait()
        await connection.close()
        await run_task

    server = await websockets.serve(server_handler, "localhost", PORT)
    await client_handler(f"ws://localhost:{PORT}")

    server.close()
    await server.wait_closed()


@pytest.mark.asyncio
async def test_rejecting_ids():
    """Tests that messages with incorrect IDs are discarded after a reset."""
    id_ = "1"
    server_done = asyncio.Event()

    async def client_handler(uri):
        async with websockets.connect(uri) as websocket:
            connection = StreamingConnection("client", websocket)
            run_task = asyncio.create_task(connection.run())

            connection.send({"id": id_, "text": "a"})
            connection.send({"id": "2", "text": "x"})
            connection.send({"id": id_, "text": "b"})
            connection.send({"id": "2", "text": "y"})
            connection.send({"id": id_, "text": "c"})

            await server_done.wait()
            await connection.close()
            await run_task

    async def server_handler(websocket):
        connection = StreamingConnection("server", websocket)
        connection.reset(id_, propagate=False)
        run_task = asyncio.create_task(connection.run())

        received = []
        start_time = asyncio.get_event_loop().time()
        while len(received) < 3 and (asyncio.get_event_loop().time() - start_time) < TIMEOUT:
            received.extend(connection.recv())
            await asyncio.sleep(0.01)

        assert len(received) == 3
        assert ''.join([msg["text"] for msg in received]) == "abc"

        server_done.set()
        await connection.close()
        await run_task

    server = await websockets.serve(server_handler, "localhost", PORT)
    await client_handler(f"ws://localhost:{PORT}")

    server.close()
    await server.wait_closed()


@pytest.mark.asyncio
async def test_raising_stream_reset():
    """Tests that StreamReset is raised when sending with an invalid ID."""
    client_done = asyncio.Event()
    server_done = asyncio.Event()

    async def client_handler(uri):
        async with websockets.connect(uri) as websocket:
            connection = StreamingConnection("client", websocket)
            run_task = asyncio.create_task(connection.run())

            connection.reset("1")  # This also sends a RESET to the server

            # This should fail because the ID is wrong
            with pytest.raises(StreamReset) as excinfo:
                connection.send({"id": "2"})
            assert excinfo.value.id == "1"

            # This should succeed
            connection.send({"id": "1", "text": "hello"})

            await server_done.wait()
            client_done.set()
            await connection.close()
            await run_task

    async def server_handler(websocket):
        connection = StreamingConnection("server", websocket)
        run_task = asyncio.create_task(connection.run())

        # This will block until the client sends the "hello" message.
        # By the time we receive it, the server's connection will have been reset
        # by the RESET message sent from the client.
        msg = await receive_one(connection)
        assert msg['id'] == '1'
        assert connection.communication_id == '1'

        # Now, test that sending with the wrong ID fails on the server side too
        with pytest.raises(StreamReset) as excinfo:
            connection.send({"id": "2"})
        assert excinfo.value.id == "1"

        server_done.set()
        await client_done.wait()
        await connection.close()
        await run_task

    server = await websockets.serve(server_handler, "localhost", PORT)
    await client_handler(f"ws://localhost:{PORT}")

    server.close()
    await server.wait_closed()