import whisper
from pathlib import Path
from socket import create_server
from utils.audio import AudioConfig
from utils.sample import Sample
from utils.message import recv_message, send_message, recv_bytes_stream

PORT = 12345
MODEL = 'medium'
SAVE_DIR = Path(__file__).parent.resolve() / 'outputs/v0'


def initialize(sock):
    msg = recv_message(sock)
    return AudioConfig(**msg.data)


if __name__ == '__main__':
    model = whisper.load_model(MODEL, device='cuda')
    options = whisper.DecodingOptions()
    with create_server(('0.0.0.0', PORT)) as sock:
        while True:
            conn_sock, conn_addr = sock.accept()
            print(f"Connected by {conn_addr}")
            audio_config = initialize(conn_sock)
            send_message({'response': 'OK'}, conn_sock)

            bytes_ = recv_bytes_stream(conn_sock)

            sample = Sample([bytes_], audio_config)
            sample.transcribe(model, options)
            sample.save(SAVE_DIR)
            print(sample.result.text)

            response = {'text': sample.result.text}
            send_message(response, conn_sock)
            conn_sock.close()
