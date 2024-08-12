## Description

A voice application with two modes of operation:

1. A notes app for transcribing spoken notes
2. A chat app for end-to-end voice conversations with LLMs

## Setup

- Clone the model repositories to the `models` directory:
    1. Speech-to-text: https://huggingface.co/openai/whisper-medium
    2. Chat: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
    3. Text-to-Speech: https://huggingface.co/coqui/XTTS-v2

- **For the python client** the requirements in `client/requirements.txt` must be installed. For PyAudio the dev
libraries of PortAudio need to be installed (cf. `server/stt/Dockerfile`)

## Run

1. Server: Assuming Docker Compose is installed: Execute `./run.sh` in the `server` directory.
2. Python-Client: `python -m client.client` (in `voice_note` directory)
