import asyncio
import time
import websockets
from concurrent.futures import ThreadPoolExecutor
from base_server import SimplexServer
from utils.message import Message


class TTSServer(SimplexServer):
    def __init__(self, host: str, port: int):
        super().__init__(host, port)

    async def process(self) -> Message.DataDict:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            transcription = await loop.run_in_executor(pool, self.dummy_transcribe)
            async with websockets.connect('ws://localhost:12346') as chat_websocket:
                await chat_websocket.send(Message({"status": "FINISHED", "transcription": transcription}).encode())
                while True:
                    response = Message.decode(await chat_websocket.recv())
                    self.send_to_client(response)
                    if response["status"] == "FINISHED":
                        break

    def dummy_transcribe(self):
        time.sleep(1)
        return b''.join([msg["audio"] for msg in self.received]).decode()


if __name__ == '__main__':
    asyncio.run(TTSServer('0.0.0.0', '12345').serve_forever())
