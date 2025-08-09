## Description

A chat app for end-to-end voice conversations with LLMs that stores the conversation in text and audio format on the server

## Setup

- Clone the model repositories to the `models` directory:
    1. Speech-to-text: https://huggingface.co/openai/whisper-medium
    2. Chat: https://huggingface.co/Qwen/Qwen3-8B-FP8
    3. Text-to-Speech: https://huggingface.co/kyutai/tts-1.6b-en_fr, https://huggingface.co/kyutai/tts-voices

- **For the python client** the requirements in `client/requirements.txt` must be installed. For PyAudio the dev
libraries of PortAudio need to be installed (cf. `server/stt/Dockerfile`)

## Run

1. Server: Assuming Docker Compose is installed: Execute `./run.sh` in the `server` directory.
2. Python-Client: `python -m client.client` (in `voice_note` directory)
