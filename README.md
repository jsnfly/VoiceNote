## Description

A voice chat app that streams conversations with an LLM-based agent and stores them as text + audio on the server.

## Architecture

The server runs three WebSocket services, chained in a pipeline:

- **STT** receives audio from the client, transcribes it with Whisper, then forwards the transcription to Chat and relays Chat/TTS responses back to the client.
- **Chat** runs [Pi coding agent](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) in RPC mode, backed by a llama.cpp model server (gemma-4-12B). Text deltas stream from Pi to both the client (as text) and TTS (for synthesis).
- **TTS** receives text chunks from Chat, synthesizes audio with [dots.tts](https://github.com/rednote-hilab/dots.tts) (the MeanFlow-distilled `rednote-hilab/dots.tts-mf` checkpoint) using zero-shot voice cloning from a reference audio prompt, and streams 48 kHz PCM audio back through STT to the client.

All three services inherit from `BaseServer`, which manages the WebSocket connection lifecycle. The `StreamingConnection` class handles bidirectional send/recv with queues and ID-based message validation (see [Interruption Mechanism](#interruption-mechanism)).

## Message Flow

Every message carries an `id` (UUID) and `status`. The `id` ties together all messages belonging to one user utterance. A typical flow:

```
Client                        STT                           Chat                          TTS
  │                             │                             │                             │
  │── {id, INITIALIZING} ──────►│                             │                             │
  │── {id, RECORDING, audio} ──►│                             │                             │
  │── {id, RECORDING, audio} ──►│  (accumulates audio)        │                             │
  │── {id, FINISHED, audio} ───►│                             │                             │
  │                             │── {id, text} ──────────────►│                             │
  │                             │                             │── {id, text} ──────────────►│
  │                             │                             │  (streams text deltas)      │
  │                             │◄─ {id, GENERATING, text} ───│                             │
  │◄─ {id, GENERATING, text} ───│                             │── {id, GENERATING, text} ──►│
  │                             │                             │  (streaming continues...)   │
  │                             │                             │── {id, FINISHED} ──────────►│
  │                             │                             │                             │── (generates remaining audio)
  │                             │◄─ {id, GENERATING, audio} ──│─────────────────────────────│
  │◄─ {id, GENERATING, audio} ──│                             │                             │
  │                             │◄─ {id, FINISHED, audio} ────│─────────────────────────────│
  │◄─ {id, FINISHED, audio} ────│                             │                             │
```

## Interruption Mechanism

When the user starts a new recording while the assistant is still speaking, the system must abort all in-flight work and start fresh. This is handled via the `communication_id` on each `StreamingConnection`:

1. **Client** generates a new UUID, calls `connection.reset(new_id)` which clears its send/recv queues and sends `{status: RESET, id: new_id}` to STT.
2. **STT** receives the RESET, propagates `stream.reset(new_id)` to its Chat connection. Any in-flight transcription is cancelled via `StreamReset`.
3. **Chat** receives the reset. If a prompt is in-flight, the workload task is cancelled → `pi.abort()` is sent to the Pi RPC subprocess (with a 5s timeout). The reset is propagated to TTS.
4. **TTS** receives the reset. The `AsyncTTSGenerator` is restarted: the generation task is cancelled, text/audio queues are cleared, and a fresh generation loop starts.
5. At every stage, any attempt to `send()` with a stale `id` raises `StreamReset`, which cascades the reset to all downstream streams.

Messages with a stale `id` are silently discarded by `StreamingConnection._recv_to_queue()`, so old audio/text fragments never reach the client.

## Pi-Agent RPC Protocol

The Chat server spawns Pi as a subprocess (`pi --mode rpc`) and communicates via newline-delimited JSON over stdin/stdout.

**Commands sent to Pi (stdin):**

| Type | Description |
|---|---|
| `new_session` | Start a fresh conversation context |
| `abort` | Cancel the current prompt |
| `prompt` | Send a user message (`{type, id, message}`) |
| `extension_ui_response` | Auto-cancel interactive UI requests (select/confirm/input/editor) |

**Events received from Pi (stdout):**

| Type | Description |
|---|---|
| `response` | Acknowledgment of a command (`{id, success, error?}`) |
| `message_update` | Text delta from the LLM (`{assistantMessageEvent: {type: text_delta, delta}}`) |
| `agent_end` | Pi finished processing the prompt |
| `extension_ui_request` | Pi requesting user interaction (auto-cancelled in voice mode) |

The `PiRpcClient` class manages the subprocess lifecycle and provides `prompt()` as an async generator that yields events.

## Setup

Clone the repo with its submodules (the TTS service depends on a vendored copy of
[dots.tts](https://github.com/rednote-hilab/dots.tts)):

```bash
git clone --recurse-submodules <this-repo>
# or, if you already cloned without --recurse-submodules:
git submodule update --init
```

Then clone the model repositories into the `models` directory:

1. **Speech-to-text**: https://huggingface.co/openai/whisper-medium → `models/whisper-medium`
2. **Chat model**: https://huggingface.co/unsloth/gemma-4-12b-it-GGUF → `models/chat/gemma-4-12b-it-GGUF`
   Place `gemma-4-12b-it-UD-Q6_K_XL.gguf` in that directory.
3. **Text-to-speech**: https://huggingface.co/rednote-hilab/dots.tts-mf → `models/dots.tts-mf`
   The `voice_note/server/tts/dots.tts/` directory is provided by the submodule above; the TTS server imports it from there at runtime. A CUDA GPU is required. For voice cloning, place a short reference WAV at `voice_note/server/tts/sample.wav` (the server reads it from there).

## Run (Docker)

```bash
cd voice_note/server
./run.sh
```

This starts all four containers (stt, chat, llamacpp, tts) via Docker Compose. The chat service runs Pi in RPC mode with coding tools enabled and the repo mounted at `/workspace`.

## Run (Host)

Requires **Node.js >= 22** (for Pi). Install Pi once:

```bash
cd voice_note/server && npm install
```

Start the llama.cpp model server (simplest via Compose):

```bash
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
| `CHAT_AGENT_CWD` | project root | Working directory for Pi's tools |
| `LLAMACPP_BASE_URL` | `http://localhost:8080/v1` | llama.cpp OpenAI-compatible API URL |
| `PI_COMMAND` | auto-detected | Override the Pi executable path |
| `TTS_URI` | `ws://localhost:12347` | TTS websocket URI |
| `CHAT_URI` | `ws://localhost:12346` | Chat server WebSocket URI (for STT) |
| `DEBUG` | (unset) | Set to enable per-connection debug log files in `logs/` |

The rightmost 2 columns represent Docker defaults. On the host, the URIs default to `localhost` instead
of the Docker service names.

### Pi Agent Configuration

The Pi coding agent reads its configuration from the working directory (`CHAT_AGENT_CWD`):

- **`models.json`** in `voice_note/pi-agent/` defines the model provider. On startup, only the
  `baseUrl` field is patched from `LLAMACPP_BASE_URL`; all other settings are read from the
  committed file. To change the model or provider, edit this file directly.
- **`.pi/APPEND_SYSTEM.md`** at the project root appends voice-output instructions to Pi's
  default system prompt. Edit this file to customize how the agent phrases its spoken responses.

### Debug Logging

Set `DEBUG=1` to enable detailed per-connection log files in `logs/`. Each connection (client,
chat, tts) writes a separate file with all sent and received messages. Useful for tracing
interruption handling and message flow.

## Client

```bash
python -m client.client
```

Run from the `voice_note` directory. Install requirements from `client/requirements.txt` first (PyAudio requires PortAudio dev libraries).

## Tests

```bash
pip install pytest pytest-asyncio websockets
cd voice_note && pytest -q
```

## Android App

Build a debug APK from the `android_app/` directory (requires a JDK and the
Android SDK, typically via `ANDROID_HOME`):

```bash
cd android_app && ./gradlew assembleDebug
```

The unsigned debug APK is written to
`android_app/app/build/outputs/apk/debug/app-debug.apk` and can be installed
with `adb install`. The `app/build.gradle` does not configure a release
signing key, so use `assembleRelease` only after adding a `signingConfig` to
the `release` build type.
