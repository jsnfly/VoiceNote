import asyncio
import pytest
import websockets
from server.utils.streaming_connection import StreamingConnection, StreamReset, POLL_INTERVAL

PORT = 7755
TIMEOUT = 1.0  # Seconds.


@pytest.mark.asyncio
async def test_send_and_receive():
    data = {"binary_text": "abcd".encode(), "text": "abcd"}
    msg = {"id": "1234", **data, "nested": data}
    client_received = asyncio.Event()

    async def _check_receive(connection):
        while True:
            received = connection.recv()
            if received:
                assert received[0] == msg
                break
            await asyncio.sleep(0.1)

    async def client_handler(uri):
        async with websockets.connect(uri) as websocket:
            connection = StreamingConnection(websocket)
            run_task = asyncio.create_task(connection.run())

            connection.send(msg)
            await _check_receive(connection)
            client_received.set()

            await connection.close()
            await run_task

    async def server_handler(websocket):
        connection = StreamingConnection(websocket)
        run_task = asyncio.create_task(connection.run())

        await _check_receive(connection)
        connection.send(msg)

        await client_received.wait()
        await connection.close()
        await run_task

    server = await websockets.serve(server_handler, "localhost", PORT)
    await client_handler(f"ws://localhost:{PORT}")

    server.close()
    await server.wait_closed()


@pytest.mark.asyncio
async def test_rejecting_ids():
    server_received = asyncio.Event()
    id_ = "1"

    async def client_handler(uri):
        async with websockets.connect(uri) as websocket:
            connection = StreamingConnection(websocket)
            run_task = asyncio.create_task(connection.run())

            connection.send({"id": id_, "text": "a"})
            connection.send({"id": "2", "text": "x"})
            connection.send({"id": id_, "text": "b"})
            connection.send({"id": "2", "text": "x"})
            connection.send({"id": id_, "text": "c"})

            await server_received.wait()
            await connection.close()
            await run_task

    async def server_handler(websocket):
        connection = StreamingConnection(websocket)
        connection.reset(id_, propagate=False)
        run_task = asyncio.create_task(connection.run())

        received = []
        start_time = asyncio.get_event_loop().time()
        while len(received) < 3 and (asyncio.get_event_loop().time() - start_time) < TIMEOUT:
            received += connection.recv()
            await asyncio.sleep(0.1)

        assert ''.join([msg["text"] for msg in received]) == "abc"

        server_received.set()
        await connection.close()
        await run_task

    server = await websockets.serve(server_handler, "localhost", PORT)
    await client_handler(f"ws://localhost:{PORT}")

    server.close()
    await server.wait_closed()


@pytest.mark.asyncio
async def test_raising_stream_reset():
    client_sent = asyncio.Event()
    server_sent = asyncio.Event()

    async def client_handler(uri):
        async with websockets.connect(uri) as websocket:
            connection = StreamingConnection(websocket)
            run_task = asyncio.create_task(connection.run())

            connection.reset("1")
            with pytest.raises(StreamReset) as excinfo:
                connection.send({"id": "2"})

            assert excinfo.value.id == "1"

            client_sent.set()
            await server_sent.wait()
            await connection.close()
            await run_task

    async def server_handler(websocket):
        connection = StreamingConnection(websocket)
        run_task = asyncio.create_task(connection.run())
        await client_sent.wait()

        # Have to wait a bit for the reset message to be received.
        await asyncio.sleep(POLL_INTERVAL * 5)

        with pytest.raises(StreamReset) as excinfo:
            connection.send({"id": "2"})

        assert excinfo.value.id == "1"

        server_sent.set()
        await connection.close()
        await run_task

    server = await websockets.serve(server_handler, "localhost", PORT)
    await client_handler(f"ws://localhost:{PORT}")

    server.close()
    await server.wait_closed()
