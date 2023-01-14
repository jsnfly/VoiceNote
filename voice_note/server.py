import argparse
import time
import whisper
from socket import create_server
from time import sleep
from utils import audio, Sample, recv_messages, send_message
from utils.misc import round_to_nearest_appropriate_number
from actions import Delete, Replay
from server_config import SAVE_DIR, SAMPLE_OVERLAP, MAXIMUM_PREDICTION_FREQ


def initialize(sock):
    messages, _ = recv_messages(sock)
    audio_config = messages[0]
    send_message({'response': 'OK'}, sock)

    actions = [Delete(SAVE_DIR), Replay(SAVE_DIR)]
    return audio_config, actions


def prediction_loop(sock, audio_config, actions, save_predictions):
    sample = Sample([], audio_config['rate'])
    while True:
        start = time.time()
        messages, bytes_ = recv_messages(sock)
        assert len(messages) == 0, "Currently no messages should come from client"
        if len(bytes_) == 0:
            break

        sample.append(bytes_)

        predict(sample)
        if sample.is_finished or sample.is_empty:
            sample = finish_sample(sample, audio_config, sock, actions, save_predictions)
        end = time.time()
        if (diff := 1 / MAXIMUM_PREDICTION_FREQ - (end - start)) > 0:
            sleep(diff)


def predict(sample):
    sample.transcribe(model, options)
    if not sample.is_empty:
        print(sample.result.text, end="\r")


def finish_sample(sample, audio_config, sock, actions, save_predictions=True):
    sample_size = audio.get_sample_size(audio_config['format'])

    if not sample.is_empty:
        response = apply_actions(actions, sample)
        response['text'] = sample.result.text
        send_message(response, sock)

        if save_predictions and not response.get('skip_saving', False):
            sample.save(SAVE_DIR, audio_config['channels'], sample_size)
        print("\nFinished: ", sample.result.text)

    bytes_per_second = audio_config['rate'] * sample_size
    overlap_bytes = round_to_nearest_appropriate_number(SAMPLE_OVERLAP * bytes_per_second, sample_size)
    initial_fragment = b''.join(sample.fragments)[-overlap_bytes:]
    return Sample([initial_fragment], audio_config['rate'])


def apply_actions(actions, sample):
    response = {}
    for action in actions:
        response = action(sample.result, response)
    return response


def main(port, save_predictions):
    with create_server(('0.0.0.0', port)) as sock:
        while True:
            conn_sock, conn_addr = sock.accept()
            with conn_sock:
                print(f"Connected by {conn_addr}")
                conn_sock.setblocking(0)
                audio_config, actions = initialize(conn_sock)
                prediction_loop(conn_sock, audio_config, actions, save_predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=12345, type=int)
    parser.add_argument("--no-saving", action="store_true")
    parser.add_argument("--lang")
    args = parser.parse_args()

    model = whisper.load_model('base', device='cuda')
    options = whisper.DecodingOptions(language=args.lang)

    main(args.port, not args.no_saving)
