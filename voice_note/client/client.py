import asyncio
import pyaudio
import PySimpleGUI as sg
from uuid import uuid4
import websockets
from functools import lru_cache
from server.utils.audio import audio
from server.utils.streaming_connection import StreamingConnection, POLL_INTERVAL

INPUT_DEVICE_INDEX = None
AUDIO_FORMAT = pyaudio.paInt16  # https://en.wikipedia.org/wiki/Audio_bit_depth
NUM_CHANNELS = 1  # Number of audio channels


async def start_recording(connection, input_device_index, values):
    id_ = str(uuid4())
    connection.reset(id_)
    connection.send({
        'audio_config': get_audio_config(input_device_index),
        'chat_mode': values['chat_mode'],
        'id': id_,
        'status': 'INITIALIZING',
        'topic': values['topic']
    })

    def _callback(in_data, *args):
        try:
            connection.send({'audio': in_data, 'id': id_, 'status': 'RECORDING'})
            return None, pyaudio.paContinue
        except BrokenPipeError:
            return None, pyaudio.paComplete

    stream = audio.open(
        **get_audio_config(input_device_index),
        input=True,
        frames_per_buffer=pyaudio.paFramesPerBufferUnspecified,
        input_device_index=input_device_index,
        stream_callback=_callback
    )
    return stream


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


def stop_recording(connection, stream):
    if stream is not None:
        stream.stop_stream()
        stream.close()
        connection.send({'audio': b'', 'id': connection.communication_id, 'status': 'FINISHED'})


def start_playback(config, data):
    pointer = 0
    bytes_per_sample = audio.get_sample_size(config['format'])

    def _callback(_, frame_count, *args):
        nonlocal pointer

        end_pointer = pointer + frame_count * config['channels'] * bytes_per_sample
        chunk = data[pointer:end_pointer]
        pointer = end_pointer

        return chunk, pyaudio.paContinue

    stream = audio.open(
        format=config['format'],
        channels=config['channels'],
        rate=config['rate'],
        output=True,
        stream_callback=_callback
    )
    return stream


def stop_playback(stream):
    if stream is not None:
        stream.stop_stream()
        stream.close()


def recv_messages(connection, text_messages, audio_messages):
    messages = connection.recv()
    for msg in messages:
        if 'text' in msg:
            text_messages.append(msg)
        elif 'audio' in msg:
            audio_messages.append(msg)
        else:
            assert msg['status'] == 'INITIALIZING', 'Unknown message type.'


async def main(window):
    uri = 'ws://localhost:12345'
    while True:
        try:
            websocket = await websockets.connect(uri)
            window['REC'].update(disabled=False)
            print("Connected.")
            break
        except ConnectionRefusedError:
            await asyncio.sleep(POLL_INTERVAL)

    stream = StreamingConnection(websocket)
    await asyncio.gather(stream.run(), ui(window, stream))


async def ui(window, com_stream):
    text_messages, audio_messages = [], []
    rec_stream = playback_stream = save_path = None

    while True:
        await asyncio.sleep(POLL_INTERVAL)

        event, values = window.read(timeout=0)
        window['New Chat'].update(disabled=not values['chat_mode'])

        if event == sg.WIN_CLOSED:
            await com_stream.close()
            break
        elif event == 'REC' and window['status'].get() == 'STOPPED':
            playback_stream = stop_playback(playback_stream)
            text_messages, audio_messages = [], []
            rec_stream = await start_recording(com_stream, INPUT_DEVICE_INDEX, values)
            window['status'].update('RECORDING')
            window['message'].update('')
        elif event == sg.TIMEOUT_EVENT:
            window['status'].update('STOPPED')
            rec_stream = stop_recording(com_stream, rec_stream)
            recv_messages(com_stream, text_messages, audio_messages)

            if text_messages:
                window['message'].update(window['message'].get() + ''.join([msg['text'] for msg in text_messages]))
                save_path = text_messages[0]['save_path']
                window['Delete'].update(disabled=False)
                window['Wrong'].update(disabled=False)
                text_messages = []

            # .is_stopped returns False even if .is_active is False.
            if audio_messages and (playback_stream is None or not playback_stream.is_active()):
                stop_playback(playback_stream)
                playback_stream = start_playback(audio_messages[0]['config'],
                                                 b''.join([msg['audio'] for msg in audio_messages]))
                audio_messages = []
        elif event in ['Delete', 'Wrong', 'New Chat']:
            if event != 'Wrong':
                com_stream.reset()
                window['Delete'].update(disabled=True)
                window['Wrong'].update(disabled=True)
                window['message'].update('')
            id_ = str(uuid4())
            com_stream.reset(id_)
            com_stream.send({'action': event.upper(), 'save_path': save_path, 'status': 'INITIALIZING', 'id': id_})


if __name__ == '__main__':
    elements = [
        [sg.RealtimeButton('REC', button_color='red', disabled=True)],
        [sg.Text(text='STOPPED', key='status')],
        [sg.Text(text='', size=(40, 20), key='message', background_color='#262624')],
        [
            sg.Button(button_text='Delete', disabled=True),
            sg.Button(button_text='Wrong', disabled=True),
            sg.Button(button_text='New Chat', disabled=True)
        ],
        [sg.Text(text='Topic:'), sg.Input(default_text='misc', size=(16, 1), key='topic')],
        [sg.Checkbox('Chat Mode', key='chat_mode')]
    ]
    window = sg.Window('Voice Note Client', elements, size=(400, 750), element_justification='c', finalize=True)

    asyncio.run(main(window))
