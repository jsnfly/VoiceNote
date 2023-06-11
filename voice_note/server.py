import time
import whisper
import multiprocessing
from socket import create_server
from time import sleep
from utils import audio, Sample, recv_messages, send_message
from utils.misc import round_to_nearest_appropriate_number, prepare_log_file, log_bytes
from server_config import SAVE_DIR, SAMPLE_OVERLAP, MAXIMUM_PREDICTION_FREQ

PORT = 12345
MODEL = 'medium'

BYTES_LOG_FILE = 'logs/server_bytes.log'
prepare_log_file(BYTES_LOG_FILE)


def initialize(sock):
    messages, _ = recv_messages(sock)
    audio_config = messages[0]
    send_message({'response': 'OK'}, sock)
    return audio_config


def prediction_loop(audio_config, pipe):
    model = whisper.load_model(MODEL, device='cuda')
    options = whisper.DecodingOptions()
    sample = Sample([], audio_config['rate'])
    while True:
        start = time.time()
        try:
            # TODO: own method?
            bytes_ = pipe.recv()
            print(f"Received {len(bytes_)} bytes.")
            sample.append(bytes_)
            while len(bytes_) > 1024:
                print(f"Received {len(bytes_)} bytes.")
                sample.append(bytes_)
                bytes_ = pipe.recv()
        except EOFError:
            print("Connection closed.")
            pipe.close()
        finally:
            predict(sample, model, options)
            if sample.is_finished or sample.is_empty:
                sample = finish_sample(sample, audio_config, None)
            end = time.time()
            if (diff := 1 / MAXIMUM_PREDICTION_FREQ - (end - start)) > 0:
                sleep(diff)


def receive_loop(sock, pipe):
    while True:
        messages, bytes_ = recv_messages(sock)  # TODO: Probably await in this method somewhere.
        if BYTES_LOG_FILE is not None:
            with open(BYTES_LOG_FILE, 'a') as f:
                log_bytes(bytes_, f)
        assert len(messages) == 0, "Currently no messages should come from client"
        print(f"Sending {len(bytes_)} bytes.")
        pipe.send(bytes_)  # TODO: await here (blocks if pipe is full)
        if len(bytes_) == 0:
            break
    pipe.close()


def predict(sample, model, options):
    sample.transcribe(model, options)
    if not sample.is_empty:
        print(sample.result.text, end="\r")

# TODO: remove `sock` argument
def finish_sample(sample, audio_config, sock):
    sample_size = audio.get_sample_size(audio_config['format'])

    if not sample.is_empty:
        response = {'text': sample.result.text}
        # send_message(response, sock)

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

                pipe_in, pipe_out = multiprocessing.Pipe()
                receive_process = multiprocessing.Process(target=receive_loop, args=(conn_sock, pipe_out))
                prediction_process = multiprocessing.Process(target=prediction_loop, args=(audio_config, pipe_in))

                receive_process.start()
                prediction_process.start()

                receive_process.join()
                prediction_process.join()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
