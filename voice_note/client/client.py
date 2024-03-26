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


async def start_recording(connection, input_device_index, values):
    connection.reset()
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


def stop_recording(connection, stream):
    stream.stop_stream()
    stream.close()
    connection.send({'status': 'FINISHED', 'audio': b''})
    return None


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
    stream.stop_stream()
    stream.close()
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

    stream = StreamingConnection(websocket)
    await asyncio.gather(stream.run(), ui(window, stream))


async def ui(window, com_stream):
    rec_stream = playback_stream = playback_config = save_path = None
    playback_bytes = b''

    while True:
        await asyncio.sleep(POLL_INTERVAL)
        event, values = window.read(timeout=0)

        if event == sg.WIN_CLOSED:
            await com_stream.close()
            break

        window['New Chat'].update(disabled=not values['chat_mode'])
        if event == 'REC':
            if window['status'].get() == 'STOPPED':
                rec_stream = await start_recording(com_stream, INPUT_DEVICE_INDEX, values)
                playback_bytes = b''

                if playback_stream is not None:
                    playback_stream = stop_playback(playback_stream)

                window['status'].update('RECORDING')
                window['message'].update('')
        elif event == sg.TIMEOUT_EVENT:
            window['status'].update('STOPPED')
            if rec_stream is not None:
                rec_stream = stop_recording(com_stream, rec_stream)
                window['Delete'].update(disabled=False)
                window['Wrong'].update(disabled=False)

            audio_messages, text_messages = [], []
            messages = com_stream.recv()
            for msg in messages:
                if 'text' in msg:
                    text_messages.append(msg)
                elif 'audio' in msg:
                    if 'config' in msg:
                        playback_config = msg['config']
                    audio_messages.append(msg)
                else:
                    assert msg['status'] == 'INITIALIZING', 'Unknown message type.'
            window['message'].update(window['message'].get() + ''.join([msg['text'] for msg in text_messages]))

            if text_messages:
                save_path = text_messages[0]['save_path']

            if audio_messages:
                playback_bytes = b''.join([playback_bytes, *[msg['audio'] for msg in audio_messages]])

            if playback_bytes:
                if playback_stream is None:
                    playback_stream = start_playback(playback_config, playback_bytes)
                    playback_bytes = b''

                # .is_stopped returns False even if .is_active is False.
                elif not playback_stream.is_active():
                    stop_playback(playback_stream)
                    playback_stream = start_playback(playback_config, playback_bytes)
                    playback_bytes = b''

        elif event in ['Delete', 'Wrong', 'New Chat']:
            com_stream.send({'action': event.upper(), 'save_path': save_path, 'status': 'ACTION'})

            # TODO: reset? (crash when using new chat when answer is not finished and starting recording after that)
            if event != 'Wrong':
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
