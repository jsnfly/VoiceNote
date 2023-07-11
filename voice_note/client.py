import pyaudio
import socket
import time
import PySimpleGUI as sg
from functools import lru_cache, partial
from utils.audio import audio
from utils.message import recv_message, send_message

HOST = '0.0.0.0'
PORT = 12345
INPUT_DEVICE_INDEX = None
AUDIO_FORMAT = pyaudio.paInt16  # https://en.wikipedia.org/wiki/Audio_bit_depth,
NUM_CHANNELS = 1  # Number of audio channels


def setup_connection(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            sock.connect((host, port))
            break
        except ConnectionRefusedError:
            time.sleep(0.1)
    return sock


def setup_stream(sock, input_device_index, topic):
    msg = {
        'audio_config': get_audio_config(input_device_index),
        'topic': topic
    }
    send_message(msg, sock)
    stream = audio.open(
        **get_audio_config(input_device_index),
        input=True,
        frames_per_buffer=pyaudio.paFramesPerBufferUnspecified,
        input_device_index=input_device_index,
        stream_callback=partial(callback, sock)
    )
    return stream


def callback(sock, in_data, *args):
    try:
        sock.sendall(in_data)
        return None, pyaudio.paContinue
    except BrokenPipeError:
        return None, pyaudio.paComplete


@lru_cache(maxsize=1)
def get_audio_config(input_device_index):
    if input_device_index is None:
        device_config = audio.get_default_input_device_info()
    else:
        device_config = audio.get_device_info_by_index(input_device_index)

    return {
        'format': AUDIO_FORMAT,
        'channels': NUM_CHANNELS,
        'rate': int(device_config['defaultSampleRate'])
    }


def teardown(sock, stream):
    stream.stop_stream()
    stream.close()

    sock.shutdown(1)
    msg = recv_message(sock)
    sock.close()
    return msg.data


def delete_message(response):
    sock = setup_connection(HOST, PORT)
    send_message({'action': 'delete', 'save_path': response['save_path']}, sock)
    sock.close()


if __name__ == "__main__":
    sock = stream = response = None

    elements = [
        [sg.RealtimeButton("REC", button_color="red")],
        [sg.Text(text="STOPPED", key="status")],
        [sg.Text(text="", size=(40, 20), key="message", background_color='#262624')],
        [sg.Button(button_text="Delete", disabled=True)],
        [sg.Text(text="Topic:"), sg.Input(default_text="misc", size=(16, 1), key="topic")]
    ]
    window = sg.Window("Voice Note Client", elements, size=(400, 750), element_justification="c", finalize=True)
    while True:
        event, _ = window.read(timeout=100)
        if event == sg.WIN_CLOSED:
            break
        elif event == sg.TIMEOUT_EVENT:
            if window["status"].get() == "RECORDING":
                response = teardown(sock, stream)
                window["message"].update(response["text"])
                window["Delete"].update(disabled=False)
            window["status"].update("STOPPED")
        elif event == 'Delete':
            delete_message(response)
            window["Delete"].update(disabled=True)
            window["message"].update("")
        else:
            if window["status"].get() == "STOPPED":
                window["status"].update("CONNECTING...")
                sock = setup_connection(HOST, PORT)
                stream = setup_stream(sock, INPUT_DEVICE_INDEX, window["topic"].get())
            window["status"].update("RECORDING")
