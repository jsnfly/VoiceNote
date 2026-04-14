## Description

A chat app for end-to-end voice conversations with LLMs that stores the conversation in text and audio format on the server

## Setup

- Clone the model repositories to the `models` directory:
    1. Speech-to-text: https://huggingface.co/openai/whisper-medium
    2. Chat: https://huggingface.co/unsloth/Qwen3.5-9B-GGUF and place `Qwen3.5-9B-Q8_0.gguf` in `models/chat/Qwen3.5-9B-GGUF`
       Also download `mmproj-F16.gguf` into the same directory to keep multimodal support available.
    3. Text-to-Speech: https://huggingface.co/kyutai/tts-1.6b-en_fr, https://huggingface.co/kyutai/tts-voices

- **For the python client** the requirements in `client/requirements.txt` must be installed. For PyAudio the dev
libraries of PortAudio need to be installed (cf. `server/stt/Dockerfile`)

## Run

1. Server: Assuming Docker Compose is installed: Execute `./run.sh` in the `server` directory.
2. Python-Client: `python -m client.client` (in `voice_note` directory)

## Tests

The current automated tests are async integration tests around the websocket streaming layer and fake server pipelines.

Install the minimal test dependencies into your existing Python environment:

```bash
python -m pip install pytest pytest-asyncio websockets
```

Run the current test suite from the `voice_note` directory:

```bash
pytest -q tests/test_streaming_connection.py
```

If more tests are added later, you can run all tests with:

```bash
pytest -q
```
