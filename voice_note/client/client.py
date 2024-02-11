import asyncio
import pyaudio
import PySimpleGUI as sg
import websockets
from functools import lru_cache
from server.utils.audio import audio
from server.utils.streaming_connection import StreamingConnection, POLL_INTERVAL

INPUT_DEVICE_INDEX = None
AUDIO_FORMAT = pyaudio.paInt16  # https://en.wikipedia.org/wiki/Audio_bit_depth,
NUM_CHANNELS = 1  # Number of audio channels


def start_recording(connection, input_device_index, values):
    msg = {
        'audio_config': get_audio_config(input_device_index),
        'chat_mode': values['chat_mode'],
        'status': 'INITIALIZING',
        'topic': values['topic']
    }
    connection.send(msg)

    def _callback(in_data, *args):
        try:
            connection.send({'status': 'RECORDING', 'audio': in_data})
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


def start_playback(config, callback):
    return audio.open(
        format=config['format'],
        channels=config['channels'],
        rate=config['rate'],
        output=True,
        stream_callback=callback
    )


def stop_recording(connection, stream):
    stream.stop_stream()
    stream.close()
    connection.send({'status': 'FINISHED', 'audio': b''})
    return None


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

    connection = StreamingConnection(websocket)
    await asyncio.gather(connection.run(), ui(window, connection))


async def ui(window, connection):
    received_bytes = b''
    playback_stream = None
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        event, values = window.read(timeout=0)

        if event == sg.WIN_CLOSED:
            await connection.close()
            break

        # TODO: never True?
        if playback_stream is not None and playback_stream.is_stopped():
            playback_stream = None
            received_bytes = b''

        window['New Chat'].update(disabled=not values['chat_mode'])
        if event == 'REC':
            if window['status'].get() == 'STOPPED':
                rec_stream = start_recording(connection, INPUT_DEVICE_INDEX, values)
                window['status'].update('RECORDING')
                window['message'].update('')
        elif event == sg.TIMEOUT_EVENT:
            if window['status'].get() == 'RECORDING':
                if rec_stream is not None:
                    rec_stream = stop_recording(connection, rec_stream)

                messages = connection.recv()
                window['message'].update(window['message'].get() + ''.join([msg.get('text', '') for msg in messages]))

                received_bytes = b''.join([received_bytes, *[msg.get('audio', b'') for msg in messages]])

                if received_bytes and playback_stream is None:
                    pointer = 0
                    config = next(msg for msg in messages if 'audio' in msg)['config']
                    bytes_per_sample = audio.get_sample_size(config['format'])

                    def _callback(_, frame_count, *args):
                        nonlocal pointer

                        end_pointer = pointer + frame_count * config['channels'] * bytes_per_sample
                        chunk = received_bytes[pointer:end_pointer]
                        pointer = end_pointer
                        return chunk, pyaudio.paContinue

                    playback_stream = start_playback(config, _callback)

                if any(msg['status'] == 'FINISHED' for msg in messages):
                    save_path = messages[-1]['save_path']
                    window['status'].update('STOPPED')
                    window['Delete'].update(disabled=False)
                    window['Wrong'].update(disabled=False)
        elif event in ['Delete', 'Wrong']:
            connection.send({'action': event.upper(), 'save_path': save_path, 'status': 'ACTION'})
            if event == 'Delete':
                window['Delete'].update(disabled=True)
                window['Wrong'].update(disabled=True)
                window['message'].update('')

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
