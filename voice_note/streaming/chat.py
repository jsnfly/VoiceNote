import asyncio
import time
from base_server import BaseServer, POLL_INTERVAL
from utils.message import Message


class ChatServer(BaseServer):

    async def handle_workload(self):
        while True:
            received = self.recv_from_client()
            if not received:
                await asyncio.sleep(POLL_INTERVAL)
            else:
                # TODO: handle multiple messages.
                await self.run_blocking_function_in_thread(self.dummy_generate, [received[0]])

    def dummy_generate(self, message: Message) -> None:
        for token in f"You said: '{message['transcription']}'".split():
            time.sleep(0.5)
            self.send_to_client({"status": "RESPONDING", "token": token})
        self.send_to_client({"status": "FINISHED", "token": "."})


if __name__ == '__main__':
    asyncio.run(ChatServer('0.0.0.0', '12346').serve_forever())
