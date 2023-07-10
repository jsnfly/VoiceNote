import whisper
from pathlib import Path
from socket import create_server
from utils.audio import AudioConfig
from utils.sample import Sample
from utils.message import recv_message, send_message, recv_bytes_stream

PORT = 12345
MODEL = 'medium'
SAVE_DIR = Path(__file__).parent.resolve() / 'outputs/v0'


def handle_deletion(path):
    save_path = Path(init_msg['Delete'])
    for file in save_path.iterdir():
        file.unlink()
    save_path.rmdir()
    print(f"Deleted {save_path}.")


def handle_audio_stream(audio_config, sock):
    bytes_ = recv_bytes_stream(sock)

    sample = Sample([bytes_], audio_config)
    sample.transcribe(model, options)
    save_path = sample.save(SAVE_DIR)
    print(sample.result.text)

    response = {'text': sample.result.text, 'save_path': str(save_path)}
    send_message(response, sock)
    sock.close()


if __name__ == '__main__':
    model = whisper.load_model(MODEL, device='cuda')
    options = whisper.DecodingOptions()
    with create_server(('0.0.0.0', PORT)) as sock:
        while True:
            conn_sock, conn_addr = sock.accept()
            print(f"Connected by {conn_addr}")
            init_msg = recv_message(conn_sock)
            send_message({'response': 'OK'}, conn_sock)
            if 'Delete' in init_msg:
                handle_deletion(init_msg['Delete'])
                send_message({'response': 'OK'}, conn_sock)
            else:
                handle_audio_stream(AudioConfig(**init_msg.data), conn_sock)
