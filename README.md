## Description

A chat app for end-to-end voice conversations with LLMs that stores the conversation in text and audio format on the server.

## Architecture

The server has three services:

- **STT** - Speech-to-text using Whisper (transformers)
- **Chat** - Conversational agent using [Pi coding agent](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) in RPC mode, backed by a llama.cpp model server
- **TTS** - Text-to-speech using Kyutai TTS

## Setup

Clone the model repositories into the `models` directory:

1. **Speech-to-text**: https://huggingface.co/openai/whisper-medium → `models/whisper-medium`
2. **Chat model**: https://huggingface.co/unsloth/Qwen3.5-9B-GGUF → `models/chat/Qwen3.5-9B-GGUF`
   Place `Qwen3.5-9B-Q8_0.gguf` in that directory. Optionally download `mmproj-F16.gguf` for multimodal support.
3. **Text-to-speech**: https://huggingface.co/kyutai/tts-1.6b-en_fr → `models/tts-1.6b-en_fr`
   and https://huggingface.co/kyutai/tts-voices → `models/tts-voices`

## Run (Docker)

```bash
cd voice_note/server
./run.sh
```

This starts all four containers (stt, chat, llamacpp, tts) via Docker Compose. The chat service runs Pi in RPC mode with coding tools enabled and the repo mounted at `/workspace`.

## Run (Host)

For running the Python servers directly on the host, you need **Node.js >= 22** (required by Pi).

Install Pi once:

```bash
cd voice_note/server
npm install
```

Start the llama.cpp model server (simplest via Compose):

```bash
cd voice_note/server
docker compose up llamacpp
```

Then start the Python servers:

```bash
python -m server.stt.stt
python -m server.chat.chat
python -m server.tts.tts
```

The chat service writes `voice_note/pi-agent/models.json` automatically on startup, pointing at `http://localhost:8080/v1`.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CHAT_AGENT_TOOLS` | `read-only` | Tool set for Pi: `read-only`, `coding`, `none`, or a custom comma-separated list |
| `CHAT_AGENT_CWD` | current directory | Working directory for Pi's tools |
| `PI_MODEL` | `Qwen3.5-9B-Q8_0` | Model ID to use |
| `LLAMACPP_BASE_URL` | `http://localhost:8080/v1` | llama.cpp OpenAI-compatible API URL |
| `PI_COMMAND` | auto-detected | Override the Pi executable path |
| `TTS_URI` | `ws://localhost:12347` | TTS websocket URI |

## Client

```bash
python -m client.client
```

Run from the `voice_note` directory. Install requirements from `client/requirements.txt` first (PyAudio requires PortAudio dev libraries).

## Tests

```bash
pip install pytest pytest-asyncio websockets
cd voice_note
pytest -q
```
