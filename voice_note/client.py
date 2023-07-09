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


def setup(host, port, input_device_index):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            sock.connect((host, port))
            send_message(get_audio_config(input_device_index), sock)
            msg = recv_message(sock)
            assert msg['response'] == 'OK'
            break
        except ConnectionRefusedError:
            time.sleep(0.2)

    stream = audio.open(
            **get_audio_config(input_device_index),
            input=True,
            frames_per_buffer=pyaudio.paFramesPerBufferUnspecified,
            input_device_index=input_device_index,
            stream_callback=partial(callback, sock)
        )

    return sock, stream


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


def callback(sock, in_data, *args):
    try:
        sock.sendall(in_data)
        return None, pyaudio.paContinue
    except BrokenPipeError:
        return None, pyaudio.paComplete


def teardown(sock, stream):
    stream.stop_stream()
    stream.close()

    sock.shutdown(1)
    msg = recv_message(sock)
    window['message'].update(msg["text"])
    sock.close()


if __name__ == "__main__":
    sock = stream = None

    elements = [
        [sg.RealtimeButton("REC", button_color="red")],
        [sg.Text(text="STOPPED", key="status")],
        [sg.Text(text="", size=(40, 20), key="message", background_color='#262624')]
    ]
    window = sg.Window("Voice Note Client", elements, size=(400, 750), element_justification="c", finalize=True)
    while True:
        event, _ = window.read(timeout=100)
        if event == sg.WIN_CLOSED:
            break
        elif event == sg.TIMEOUT_EVENT:
            if window["status"].get() == "RECORDING":
                teardown(sock, stream)
            window["status"].update("STOPPED")
        else:
            if window["status"].get() == "STOPPED":
                window["status"].update("CONNECTING...")
                sock, stream = setup(HOST, PORT, INPUT_DEVICE_INDEX)
            window["status"].update("RECORDING")
