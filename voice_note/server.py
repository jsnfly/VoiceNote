import argparse
import time
import whisper
from socket import create_server
from time import sleep
from utils import audio, Sample, recv_messages, send_message
from utils.misc import round_to_nearest_appropriate_number
from server_config import SAVE_DIR, SAMPLE_OVERLAP, MAXIMUM_PREDICTION_FREQ

PORT = 12345
MODEL = 'medium'


def initialize(sock):
    messages, _ = recv_messages(sock)
    audio_config = messages[0]
    send_message({'response': 'OK'}, sock)
    return audio_config


def prediction_loop(sock, audio_config):
    sample = Sample([], audio_config['rate'])
    while True:
        start = time.time()
        messages, bytes_ = recv_messages(sock)
        assert len(messages) == 0, "Currently no messages should come from client"
        if len(bytes_) == 0:
            # TODO: why?
            break

        sample.append(bytes_)
        predict(sample)

        if sample.is_finished or sample.is_empty:
            sample = finish_sample(sample, audio_config, sock)
        end = time.time()
        if (diff := 1 / MAXIMUM_PREDICTION_FREQ - (end - start)) > 0:
            sleep(diff)


def predict(sample):
    sample.transcribe(model, options)
    if not sample.is_empty:
        print(sample.result.text, end="\r")


def finish_sample(sample, audio_config, sock):
    sample_size = audio.get_sample_size(audio_config['format'])

    if not sample.is_empty:
        response = {'text': sample.result.text}
        send_message(response, sock)

        sample.save(SAVE_DIR, audio_config['channels'], sample_size)
        print("\nFinished: ", sample.result.text)

    bytes_per_second = audio_config['rate'] * sample_size
    overlap_bytes = round_to_nearest_appropriate_number(SAMPLE_OVERLAP * bytes_per_second, sample_size)
    initial_fragment = b''.join(sample.fragments)[-overlap_bytes:]
    return Sample([initial_fragment], audio_config['rate'])


def main():
    with create_server(('0.0.0.0', PORT)) as sock:
        while True:
            conn_sock, conn_addr = sock.accept()
            with conn_sock:
                print(f"Connected by {conn_addr}")
                conn_sock.setblocking(0)
                audio_config = initialize(conn_sock)
                prediction_loop(conn_sock, audio_config)


if __name__ == '__main__':
    model = whisper.load_model(MODEL, device='cuda')
    options = whisper.DecodingOptions()

    main()
