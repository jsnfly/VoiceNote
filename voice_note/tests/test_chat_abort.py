import asyncio

import pytest

from server.chat.chat import ChatServer
from server.utils.streaming_connection import POLL_INTERVAL, StreamReset

ABORT_TIMEOUT = 2.0


class FakeStream:
    """Mimics StreamingConnection.send/recv semantics without a websocket."""

    def __init__(self, communication_id=None):
        self.communication_id = communication_id
        self.sent = []
        self._queued = []

    def send(self, data):
        if self.communication_id is None or data.get('id') == self.communication_id:
            self.sent.append(data)
        else:
            raise StreamReset('Invalid message ID', self.communication_id)

    def recv(self):
        queued, self._queued = self._queued, []
        return queued


class FakePi:
    """Mimics PiRpcClient: prompt() holds the lock across its yields and abort() needs it."""

    def __init__(self):
        self.lock = asyncio.Lock()
        self.abort_called = False

    async def prompt(self, message):
        async with self.lock:
            while True:
                yield {
                    'type': 'message_update',
                    'assistantMessageEvent': {'type': 'text_delta', 'delta': 'hello '},
                }

    async def abort(self):
        async with self.lock:
            self.abort_called = True


def make_server(client_stream, pi=None) -> ChatServer:
    server = ChatServer.__new__(ChatServer)  # Skip __init__ (pi subprocess, config IO).
    server.pi = pi if pi is not None else FakePi()
    server.new_session_requested = False
    server.streams = {'client': client_stream}
    return server


@pytest.mark.asyncio
async def test_stream_reset_in_consumer_body_releases_prompt_lock_and_aborts():
    """Regression: interrupting mid-response used to deadlock the chat server.

    The prompt generator holds the RPC lock while suspended at a yield. A StreamReset raised
    inside the `async for` body must close the generator before aborting pi, otherwise
    `pi.abort()` waits forever for the lock and the server stops answering new requests.
    """
    # The client stream was already reset to the next request, so the first send of the
    # old workload raises StreamReset inside the consumer body.
    server = make_server(FakeStream(communication_id='new-id'))

    with pytest.raises(StreamReset):
        await asyncio.wait_for(
            server._run_workload([{'id': 'old-id', 'text': 'hi'}]), timeout=ABORT_TIMEOUT
        )

    assert server.pi.abort_called


@pytest.mark.asyncio
async def test_cancellation_while_generator_suspended_releases_prompt_lock_and_aborts():
    server = make_server(FakeStream())

    task = asyncio.create_task(server._run_workload([{'id': 'req-1', 'text': 'hi'}]))
    await asyncio.sleep(POLL_INTERVAL * 20)  # A few events processed; generator suspended.
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=ABORT_TIMEOUT)

    assert server.pi.abort_called
