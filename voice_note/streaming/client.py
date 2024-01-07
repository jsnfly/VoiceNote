import asyncio
import websockets
from utils.message import Message


async def main():
    uri = 'ws://localhost:12345'
    async with websockets.connect(uri) as websocket:
        await websocket.send(Message({"status": "RECORDING", "audio": b"bla"}).encode())
        await websocket.send(Message({"status": "RECORDING", "audio": b"bla"}).encode())
        await websocket.send(Message({"status": "FINISHED", "audio": b"bla"}).encode())

        while True:
            msg = Message.decode(await websocket.recv())
            print(msg)
            if msg["status"] == "FINISHED":
                break

asyncio.run(main())
