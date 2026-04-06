import asyncio
import contextlib
import socket

import pytest
import websockets

from server.base_server import BaseServer
from server.utils.message import Message
from server.utils.streaming_connection import POLL_INTERVAL, StreamReset, StreamingConnection


HOST = "127.0.0.1"
TIMEOUT = 2.0


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


async def start_server(server: BaseServer) -> asyncio.Task:
    task = asyncio.create_task(server.serve_forever())
    await asyncio.sleep(POLL_INTERVAL * 10)
    return task


async def stop_task(task: asyncio.Task) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def open_stream(uri: str, name: str) -> tuple[StreamingConnection, asyncio.Task]:
    websocket = await websockets.connect(uri)
    connection = StreamingConnection(name, websocket)
    run_task = asyncio.create_task(connection.run())
    return connection, run_task


async def close_stream(connection: StreamingConnection, run_task: asyncio.Task) -> None:
    with contextlib.suppress(Exception):
        await connection.close()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(run_task, timeout=TIMEOUT)


async def collect_messages(connection: StreamingConnection, done, timeout: float = TIMEOUT):
    messages = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        messages.extend(connection.recv())
        if done(messages):
            return messages
        await asyncio.sleep(POLL_INTERVAL)
    raise AssertionError(f"Timed out waiting for messages. Received: {messages}")


async def wait_for(predicate, timeout: float = TIMEOUT) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(POLL_INTERVAL)
    raise AssertionError("Timed out waiting for condition")


def split_text(text: str) -> list[str]:
    midpoint = max(1, len(text) // 2)
    return [text[:midpoint], text[midpoint:]]


class CollectingIngressServer(BaseServer):
    def __init__(self, host: str, port: int, chat_uri: str):
        super().__init__("ingress", host, port)
        self.connections = {"chat": chat_uri}
        self.completed_requests = []

    def _get_cutoff_idx(self, received):
        return next((idx for idx, msg in enumerate(received) if msg["status"] == "FINISHED"), -1) + 1

    async def _run_workload(self, received):
        request_id = received[0]["id"]
        text = "".join(msg.get("text", "") for msg in received)
        self.completed_requests.append((request_id, text))
        self.streams["chat"].send({"id": request_id, "text": text, "status": "FINISHED"})

        while True:
            for message in self.streams["chat"].recv():
                self.streams["client"].send(message)
                if message["status"] == "FINISHED":
                    return
            await asyncio.sleep(POLL_INTERVAL)


class ChatLikeServer(BaseServer):
    def __init__(self, host: str, port: int, tts_uri: str):
        super().__init__("chat_like", host, port)
        self.connections = {"tts": tts_uri}
        self.requests = []

    def _get_cutoff_idx(self, received):
        return int(bool(received))

    async def _run_workload(self, received):
        request = received[0]
        request_id = request["id"]
        transformed_text = request["text"].upper()
        self.requests.append((request_id, transformed_text))

        for index, chunk in enumerate(split_text(transformed_text)):
            if not chunk:
                continue
            status = "FINISHED" if index == len(split_text(transformed_text)) - 1 else "GENERATING"
            self.streams["tts"].send({"id": request_id, "text": chunk, "status": status})
            forwarded = False
            while not forwarded:
                for message in self.streams["tts"].recv():
                    self.streams["client"].send(message)
                    if message["audio"] == chunk.encode():
                        forwarded = True
                await asyncio.sleep(POLL_INTERVAL)


class TTSLikeServer(BaseServer):
    def __init__(self, host: str, port: int):
        super().__init__("tts_like", host, port)
        self.received = []

    def _get_cutoff_idx(self, received):
        return int(bool(received))

    async def _run_workload(self, received):
        request = received[0]
        self.received.append(request)
        await asyncio.sleep(POLL_INTERVAL * 2)
        self.streams["client"].send({
            "id": request["id"],
            "audio": request["text"].encode(),
            "text": request["text"],
            "status": request["status"],
            "config": {"channels": 1, "rate": 16000},
        })


class InterruptibleServer(BaseServer):
    def __init__(self, host: str, port: int, observer_uri: str):
        super().__init__("interruptible", host, port)
        self.connections = {"observer": observer_uri}
        self.started_ids = []
        self.interrupted_ids = []
        self.cancelled_ids = []
        self.completed_ids = []

    def _get_cutoff_idx(self, received):
        return int(bool(received))

    async def _run_workload(self, received):
        request_id = received[0]["id"]
        self.started_ids.append(request_id)
        try:
            for index in range(4):
                self.streams["client"].send({
                    "id": request_id,
                    "text": f"{request_id}-chunk-{index}",
                    "status": "FINISHED" if index == 3 else "GENERATING",
                })
                await asyncio.sleep(POLL_INTERVAL * 12)
            self.completed_ids.append(request_id)
        except StreamReset:
            self.interrupted_ids.append(request_id)
            raise
        except asyncio.CancelledError:
            self.cancelled_ids.append(request_id)
            raise


class LatestWinsServer(BaseServer):
    def __init__(self, host: str, port: int):
        super().__init__("latest_wins", host, port)
        self.completed_requests = []

    def _get_cutoff_idx(self, received):
        return next((idx for idx, msg in enumerate(received) if msg["status"] == "FINISHED"), -1) + 1

    async def _run_workload(self, received):
        request_id = received[0]["id"]
        text = "".join(msg.get("text", "") for msg in received)
        self.completed_requests.append((request_id, text))
        self.streams["client"].send({"id": request_id, "text": text, "status": "FINISHED"})


class ClosingResponderServer(BaseServer):
    def __init__(self, host: str, port: int):
        super().__init__("closing_responder", host, port)

    def _get_cutoff_idx(self, received):
        return int(bool(received))

    async def _run_workload(self, received):
        await self.streams["client"].close()


class ForwardingServer(BaseServer):
    def __init__(self, host: str, port: int, downstream_uri: str):
        super().__init__("forwarding", host, port)
        self.connections = {"downstream": downstream_uri}

    def _get_cutoff_idx(self, received):
        return next((idx for idx, msg in enumerate(received) if msg["status"] == "FINISHED"), -1) + 1

    async def _run_workload(self, received):
        request_id = received[0]["id"]
        text = "".join(msg.get("text", "") for msg in received)
        self.streams["downstream"].send({"id": request_id, "text": text, "status": "FINISHED"})
        while True:
            for message in self.streams["downstream"].recv():
                self.streams["client"].send(message)
                if message["status"] == "FINISHED":
                    return
            await asyncio.sleep(POLL_INTERVAL)


@pytest.mark.asyncio
async def test_streaming_connection_roundtrip_over_websocket():
    port = get_free_port()
    received_by_server = []
    server_ready = asyncio.Event()

    async def server_handler(websocket):
        connection = StreamingConnection("server_roundtrip", websocket)
        run_task = asyncio.create_task(connection.run())
        try:
            server_ready.set()
            received_by_server.extend(await collect_messages(connection, lambda msgs: len(msgs) == 1))
            connection.send(received_by_server[0])
            await asyncio.sleep(POLL_INTERVAL * 5)
        finally:
            await close_stream(connection, run_task)

    server = await websockets.serve(server_handler, HOST, port)
    client_connection = None
    client_task = None
    try:
        client_connection, client_task = await open_stream(f"ws://{HOST}:{port}", "client_roundtrip")
        await server_ready.wait()

        payload = {
            "id": "roundtrip",
            "text": "hello",
            "binary": b"\x00\x01",
            "nested": {"value": "ok", "bytes": b"abc"},
        }
        client_connection.send(payload)

        echoed = await collect_messages(client_connection, lambda msgs: len(msgs) == 1)
        assert received_by_server == [payload]
        assert echoed == [payload]
    finally:
        if client_connection is not None and client_task is not None:
            await close_stream(client_connection, client_task)
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_streaming_connection_stays_bidirectional_while_both_sides_stream():
    port = get_free_port()
    server_messages = []
    server_ready = asyncio.Event()

    async def server_handler(websocket):
        connection = StreamingConnection("server_bidi", websocket)
        run_task = asyncio.create_task(connection.run())
        try:
            server_ready.set()
            receive_task = asyncio.create_task(collect_messages(connection, lambda msgs: len(msgs) == 3))
            for idx in range(3):
                connection.send({"id": "server", "text": f"server-{idx}", "status": "GENERATING"})
                await asyncio.sleep(POLL_INTERVAL * 3)
            server_messages.extend(await receive_task)
        finally:
            await close_stream(connection, run_task)

    server = await websockets.serve(server_handler, HOST, port)
    client_connection = None
    client_task = None
    try:
        client_connection, client_task = await open_stream(f"ws://{HOST}:{port}", "client_bidi")
        await server_ready.wait()

        for idx in range(3):
            client_connection.send({"id": "client", "text": f"client-{idx}", "status": "GENERATING"})
            await asyncio.sleep(POLL_INTERVAL * 2)

        client_messages = await collect_messages(client_connection, lambda msgs: len(msgs) == 3)
        await wait_for(lambda: len(server_messages) == 3)
        assert [msg["text"] for msg in server_messages] == ["client-0", "client-1", "client-2"]
        assert [msg["text"] for msg in client_messages] == ["server-0", "server-1", "server-2"]
    finally:
        if client_connection is not None and client_task is not None:
            await close_stream(client_connection, client_task)
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_fake_stt_chat_tts_pipeline_streams_end_to_end():
    tts_port = get_free_port()
    chat_port = get_free_port()
    ingress_port = get_free_port()
    tts_server = TTSLikeServer(HOST, tts_port)
    chat_server = ChatLikeServer(HOST, chat_port, f"ws://{HOST}:{tts_port}")
    ingress_server = CollectingIngressServer(HOST, ingress_port, f"ws://{HOST}:{chat_port}")

    tts_task = await start_server(tts_server)
    chat_task = await start_server(chat_server)
    ingress_task = await start_server(ingress_server)

    client_connection = None
    client_task = None
    try:
        client_connection, client_task = await open_stream(f"ws://{HOST}:{ingress_port}", "pipeline_client")
        client_connection.send({"id": "req-1", "text": "hel", "status": "GENERATING"})
        client_connection.send({"id": "req-1", "text": "lo", "status": "FINISHED"})

        messages = await collect_messages(
            client_connection,
            lambda msgs: any(msg["status"] == "FINISHED" for msg in msgs),
            timeout=4.0,
        )

        assert ingress_server.completed_requests == [("req-1", "hello")]
        assert chat_server.requests == [("req-1", "HELLO")]
        assert [msg["text"] for msg in tts_server.received] == ["HE", "LLO"]
        assert b"".join(msg["audio"] for msg in messages) == b"HELLO"
        assert messages[-1]["status"] == "FINISHED"
    finally:
        if client_connection is not None and client_task is not None:
            await close_stream(client_connection, client_task)
        await stop_task(ingress_task)
        await stop_task(chat_task)
        await stop_task(tts_task)


@pytest.mark.asyncio
async def test_reset_interrupts_inflight_workload_and_propagates_to_dependents():
    observer_port = get_free_port()
    server_port = get_free_port()
    observer_messages = []

    async def observer_handler(websocket):
        try:
            while True:
                observer_messages.append(Message.from_data_string(await websocket.recv()).data)
        except websockets.exceptions.ConnectionClosed:
            pass

    observer_server = await websockets.serve(observer_handler, HOST, observer_port)
    interruptible_server = InterruptibleServer(HOST, server_port, f"ws://{HOST}:{observer_port}")
    server_task = await start_server(interruptible_server)

    client_connection = None
    client_task = None
    try:
        client_connection, client_task = await open_stream(f"ws://{HOST}:{server_port}", "interrupt_client")
        client_connection.send({"id": "req-1", "text": "first", "status": "FINISHED"})

        first_messages = await collect_messages(
            client_connection,
            lambda msgs: any(msg["id"] == "req-1" for msg in msgs),
        )
        assert first_messages[0]["id"] == "req-1"

        client_connection.reset("req-2")
        client_connection.send({"id": "req-2", "text": "second", "status": "FINISHED"})

        follow_up_messages = await collect_messages(
            client_connection,
            lambda msgs: any(msg["id"] == "req-2" and msg["status"] == "FINISHED" for msg in msgs),
            timeout=4.0,
        )

        await wait_for(lambda: "req-1" in interruptible_server.interrupted_ids)
        await wait_for(
            lambda: [msg for msg in observer_messages if msg.get("status") == "RESET"]
            and {msg["id"] for msg in observer_messages if msg.get("status") == "RESET"} >= {"req-1", "req-2"},
            timeout=4.0,
        )

        assert all(not (msg["id"] == "req-1" and msg["status"] == "FINISHED") for msg in follow_up_messages)
        assert any(msg["id"] == "req-2" and msg["status"] == "FINISHED" for msg in follow_up_messages)
        await wait_for(lambda: interruptible_server.completed_ids == ["req-2"])
        assert interruptible_server.interrupted_ids == ["req-1"]
        assert interruptible_server.completed_ids == ["req-2"]
    finally:
        if client_connection is not None and client_task is not None:
            await close_stream(client_connection, client_task)
        await stop_task(server_task)
        observer_server.close()
        await observer_server.wait_closed()


@pytest.mark.asyncio
async def test_new_request_id_replaces_partial_old_request():
    port = get_free_port()
    server = LatestWinsServer(HOST, port)
    server_task = await start_server(server)

    client_connection = None
    client_task = None
    try:
        client_connection, client_task = await open_stream(f"ws://{HOST}:{port}", "latest_wins_client")
        client_connection.send({"id": "req-1", "text": "old-", "status": "GENERATING"})
        client_connection.send({"id": "req-2", "text": "new", "status": "FINISHED"})

        messages = await collect_messages(
            client_connection,
            lambda msgs: any(msg["id"] == "req-2" and msg["status"] == "FINISHED" for msg in msgs),
        )

        assert server.completed_requests == [("req-2", "new")]
        assert messages == [{"id": "req-2", "text": "new", "status": "FINISHED"}]
    finally:
        if client_connection is not None and client_task is not None:
            await close_stream(client_connection, client_task)
        await stop_task(server_task)


@pytest.mark.asyncio
async def test_server_retries_dependency_until_downstream_starts():
    tts_port = get_free_port()
    chat_port = get_free_port()
    ingress_port = get_free_port()
    ingress_server = CollectingIngressServer(HOST, ingress_port, f"ws://{HOST}:{chat_port}")
    ingress_task = await start_server(ingress_server)

    client_connection = None
    client_task = None
    chat_task = None
    tts_task = None
    try:
        client_connection, client_task = await open_stream(f"ws://{HOST}:{ingress_port}", "late_dependency_client")
        await asyncio.sleep(POLL_INTERVAL * 20)

        tts_server = TTSLikeServer(HOST, tts_port)
        chat_server = ChatLikeServer(HOST, chat_port, f"ws://{HOST}:{tts_port}")
        tts_task = await start_server(tts_server)
        chat_task = await start_server(chat_server)

        client_connection.send({"id": "late-1", "text": "retry", "status": "FINISHED"})
        messages = await collect_messages(
            client_connection,
            lambda msgs: any(msg["status"] == "FINISHED" for msg in msgs),
            timeout=4.0,
        )

        assert ingress_server.completed_requests == [("late-1", "retry")]
        assert b"".join(msg["audio"] for msg in messages) == b"RETRY"
    finally:
        if client_connection is not None and client_task is not None:
            await close_stream(client_connection, client_task)
        if chat_task is not None:
            await stop_task(chat_task)
        if tts_task is not None:
            await stop_task(tts_task)
        await stop_task(ingress_task)


@pytest.mark.asyncio
async def test_client_disconnect_or_closed_stream_ends_session_cleanly():
    leaf_port = get_free_port()
    main_port = get_free_port()
    leaf_server = ClosingResponderServer(HOST, leaf_port)
    main_server = ForwardingServer(HOST, main_port, f"ws://{HOST}:{leaf_port}")

    leaf_task = await start_server(leaf_server)
    main_task = await start_server(main_server)

    client_connection = None
    client_task = None
    try:
        client_connection, client_task = await open_stream(f"ws://{HOST}:{main_port}", "disconnect_client")
        client_connection.send({"id": "bye", "text": "stop", "status": "FINISHED"})

        await wait_for(lambda: client_connection.closed, timeout=4.0)
        with pytest.raises(ConnectionError):
            client_connection.recv()
    finally:
        if client_connection is not None and client_task is not None:
            await close_stream(client_connection, client_task)
        await stop_task(main_task)
        await stop_task(leaf_task)
