import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from base_server import SimplexServer
from utils.message import Message


class ChatServer(SimplexServer):

    async def process(self) -> Message.DataDict:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, self.dummy_generate)

    def dummy_generate(self):
        for token in ["This", "is", "the", "response"]:
            time.sleep(0.5)
            self.send_to_client({"status": "RESPONDING", "token": token})
        self.send_to_client({"status": "FINISHED", "token": "."})


if __name__ == '__main__':
    asyncio.run(ChatServer('0.0.0.0', '12346').serve_forever())
