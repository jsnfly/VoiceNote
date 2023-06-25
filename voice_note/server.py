import time
import whisper
from dataclasses import dataclass, field
from threading import Thread
from queue import Queue
from socket import create_server
from utils import audio, Sample, recv_messages, send_message
from utils.misc import round_to_nearest_appropriate_number, prepare_log_file, log_bytes
from server_config import SAVE_DIR, SAMPLE_OVERLAP, MAXIMUM_PREDICTION_FREQ

PORT = 12345
MODEL = 'medium'
BYTES_LOG_FILE = 'logs/server_bytes.log'


def initialize(sock):
    messages, _ = recv_messages(sock)
    audio_config = _parse_audio_config(messages[0])
    return audio_config


def _parse_audio_config(audio_config):

    @dataclass
    class AudioConfig:
        format: int
        channels: int
        rate: int
        sample_size: int = field(init=False)
        bytes_per_second: int = field(init=False)

        def __post_init__(self):
            self.sample_size = audio.get_sample_size(self.format)
            self.bytes_per_second = self.rate * self.sample_size

    return AudioConfig(**audio_config.data)


def communication_loop(queue, sock):
    while True:
        messages, bytes_ = recv_messages(sock)
        assert len(messages) == 0, "Currently no messages should come from client"
        if BYTES_LOG_FILE is not None:
            with open(BYTES_LOG_FILE, 'a') as f:
                log_bytes(bytes_, f)
        queue.put(bytes_)


def prediction_loop(queue, audio_config, model, options):
    sample = Sample([], audio_config.rate, audio_config.channels, audio_config.sample_size)
    while True:
        start = time.time()
        bytes_ = b''.join([queue.get() for _ in range(queue.qsize())])
        sample.append(bytes_)
        sample.transcribe(model, options)

        if sample.finished or sample.is_empty:
            sample, response = _finish_sample(sample, audio_config)
            # TODO: send response
            end = time.time()
            if (diff := 1 / MAXIMUM_PREDICTION_FREQ - (end - start)) > 0:
                time.sleep(diff)


def _finish_sample(sample, audio_config):
    if not sample.is_empty:
        response = {'text': sample.result.text}

        sample.save(SAVE_DIR)
        print("\nFinished: ", sample.result.text)
    else:
        response = {'text': ''}

    overlap_bytes = round_to_nearest_appropriate_number(SAMPLE_OVERLAP * audio_config.bytes_per_second,
                                                        audio_config.sample_size)
    initial_fragment = b''.join(sample.fragments)[-overlap_bytes:]
    sample = Sample([initial_fragment], audio_config.rate, audio_config.channels, audio_config.sample_size)

    return sample, response


if __name__ == '__main__':
    prepare_log_file(BYTES_LOG_FILE)
    queue = Queue()
    model = whisper.load_model(MODEL, device='cuda')
    options = whisper.DecodingOptions()
    with create_server(('0.0.0.0', PORT)) as sock:
        conn_sock, conn_addr = sock.accept()
        print(f"Connected by {conn_addr}")
        with conn_sock:
            conn_sock.setblocking(0)
            audio_config = initialize(conn_sock)

            prediction_thread = Thread(target=prediction_loop, args=(queue, audio_config, model, options))
            prediction_thread.start()

            communication_thread = Thread(target=communication_loop, args=(queue, conn_sock))
            communication_thread.start()

            send_message({'response': 'OK'}, conn_sock)

            prediction_thread.join()
            communication_thread.join()
