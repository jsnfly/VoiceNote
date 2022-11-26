import argparse
import pyaudio
import time
import json
import whisper
from socket import create_server
from time import sleep
from message import recv_message, send_message
from utils.sample import Sample
from utils.pyaudio import audio
from actions.replay import load_last_wavefile, trigger_condition
from server_config import SAVE_DIR, SAMPLE_OVERLAP, MAXIMUM_PREDICTION_FREQ


def initialize(sock):
    audio_config, other = recv_message(sock)
    assert len(other) == 0
    send_message({'response': 'OK'}, sock)
    return audio_config


def prediction_loop(sock, audio_config, save_predictions):
    sample = Sample([], audio_config['rate'])
    while True:
        start = time.time()
        msg, bytes_ = recv_message(sock)
        assert msg is None, "Currently no messages should come from client"
        sample.append(bytes_)
        predict(sample)
        if sample.is_finished or sample.is_empty:
            sample = finish_sample(sample, audio_config, sock, save_predictions)
        end = time.time()
        if (diff := 1 / MAXIMUM_PREDICTION_FREQ - (end - start)) > 0:
            sleep(diff)


def predict(sample):
    sample.transcribe(model, options)
    if not sample.is_empty:
        print(sample.result.text, end="\r")


def finish_sample(sample, audio_config, sock, save_predictions=True):
    if not sample.is_empty:

        # TODO: refactor
        result = {'text': sample.result.text}
        if trigger_condition(sample.result):
            result['audio'] = load_last_wavefile()
        send_message(result, sock)
        if save_predictions:
            sample.save(SAVE_DIR, audio_config['channels'], audio.get_sample_size(audio_config['format']))
        print("\nFinished: ", sample.result.text)

    bytes_per_second = audio_config['rate'] * 2  # Times 2 because each data point has 16 bits.
    initial_fragment = b''.join(sample.fragments)[-int(SAMPLE_OVERLAP * bytes_per_second):]
    return Sample([initial_fragment], audio_config['rate'])


def main(port, save_predictions):
    with create_server(('0.0.0.0', port)) as sock:
        conn_sock, conn_addr = sock.accept()
        with conn_sock:
            print(f"Connected by {conn_addr}")
            conn_sock.setblocking(0)
            audio_config = initialize(conn_sock)
            assert audio_config['format'] == pyaudio.paInt16
            prediction_loop(conn_sock, audio_config, save_predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=12345, type=int)
    parser.add_argument("--no-saving", action="store_true")
    parser.add_argument("--lang")
    args = parser.parse_args()

    model = whisper.load_model('base', device='cuda')
    options = whisper.DecodingOptions(language=args.lang)

    main(args.port, not args.no_saving)
