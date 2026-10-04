## Description

A voice chat app that streams conversations with an LLM-based agent and stores them as text + audio on the server.

## Architecture

The server runs three WebSocket services, chained in a pipeline:

- **STT** receives audio from the client, transcribes it with Whisper, then forwards the transcription to Chat and relays Chat/TTS responses back to the client.
- **Chat** runs [Pi coding agent](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) in RPC mode, backed by a llama.cpp model server (Qwen3.8-27B GGUF with draft-MTP speculative decoding). Text deltas stream from Pi to both the client (as text) and TTS (for synthesis).
- **TTS** receives text chunks from Chat, synthesizes audio with Kyutai TTS, and streams PCM audio back through STT to the client.

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
2. **STT** resets its client-facing stream, dropping pending inbound/outbound messages. Resets propagate lazily: the next `send()` carrying a stale id raises `StreamReset`, which resets STT's Chat connection and forwards `{status: RESET}` to it. (If a workload is torn down mid-transcription — e.g. on disconnect — a `StoppingCriteria` bound to the executor's cancel event stops Whisper early.)
3. **Chat** receives the reset. The in-flight prompt raises `StreamReset` at its next send attempt; the workload then closes the prompt async-generator (which releases the RPC lock — closing first is essential, since the generator holds the lock across its yields) and sends `pi.abort()` to the Pi RPC subprocess. The reset is propagated to TTS.
4. **TTS** receives the reset. The `AsyncTTSGenerator` is restarted: the generation task is cancelled, text/audio queues are cleared, and a fresh generation loop starts. The generator is additionally restarted whenever text for a new id arrives, so stale model state can never bleed into the next response.
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
| `agent_end` | Pi finished processing the prompt; `willRetry` indicates a transient error is being auto-retried |
| `extension_ui_request` | Pi requesting user interaction (auto-cancelled in voice mode) |

The `PiRpcClient` class manages the subprocess lifecycle and provides `prompt()` as an async generator that yields events.

## Setup

Clone the model repositories into the `models` directory:

1. **Speech-to-text**: https://huggingface.co/openai/whisper-large-v3-turbo → `models/whisper-large-v3-turbo`
2. **Chat model**: a Qwen3.8-27B GGUF quant with MTP draft weights, e.g. `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` → `models/chat/Qwen3.8-27B-GSQ-RCO-GGUF`
   The compose file runs llama.cpp with `--spec-type draft-mtp`, which requires the MTP draft weights in the GGUF. The `--alias` (`qwen3.8-27b`) must match the model `id` in `pi-agent/models.json`.
3. **Text-to-speech**: https://huggingface.co/kyutai/tts-1.6b-en_fr → `models/tts-1.6b-en_fr`
   and https://huggingface.co/kyutai/tts-voices → `models/tts-voices`

## Run (Docker)

```bash
cd voice_note/server
./run.sh
```

This starts all four containers (stt, chat, llamacpp, tts) via Docker Compose. The chat service runs Pi in RPC mode with coding tools enabled and the `workspace/` directory mounted at `/workspace`.

## Run (Host)

Requires **Node.js >= 22** (for Pi). Install Pi once:

```bash
cd voice_note/server && npm install
```

The chat server resolves the `pi` executable from `server/node_modules/.bin` or from `PATH` (e.g. a global install); it never installs anything implicitly.

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

On startup, the STT and TTS servers run a short warmup pass to prime CUDA before accepting connections.

The chat service writes `voice_note/pi-agent/models.json` automatically on startup, pointing at `http://localhost:8080/v1`.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CHAT_AGENT_CWD` | `workspace/` | Working directory for Pi's tools |
| `CHAT_TOOLS` | `read-only` | Tools enabled for Pi (`read-only` or `all`) |
| `LLAMACPP_BASE_URL` | `http://localhost:8080/v1` | llama.cpp API URL; written into `pi-agent/models.json` on startup |
| `TTS_URI` | `ws://localhost:12347` | TTS websocket URI |
| `CHAT_URI` | `ws://localhost:12346` | Chat server WebSocket URI (for STT) |
| `DEBUG` | (unset) | Set to enable per-connection debug log files in `logs/` |

In Docker, `compose.yml` sets `CHAT_URI`, `TTS_URI`, `LLAMACPP_BASE_URL` and `CHAT_AGENT_CWD` to the Docker service names and `/workspace`. On the host, the URIs default to `localhost` instead.

### Pi Agent Configuration

The Pi agent reads its configuration from `voice_note/pi-agent/`:

- **`models.json`** in `voice_note/pi-agent/` defines the model provider. On startup, the chat
  server rewrites its `baseUrl` from `LLAMACPP_BASE_URL`; all other settings are read from the
  committed file. To change the model or provider, edit this file directly. The model `id`
  must match the `--alias` the llama.cpp server is started with.
- **`SYSTEM.md`** in `voice_note/pi-agent/` defines the complete system prompt for the assistant,
  replacing Pi's default coding agent prompt with a concise, voice-oriented general assistant prompt.
- **`extensions/`** in `voice_note/pi-agent/` holds Pi TypeScript extensions, loaded automatically
  at startup. Currently `thinking-toggle.ts`, which adds the `set_thinking` tool for switching the
  model's thinking mode on or off (Pi is started with `--thinking off`; `set_thinking` is included
  in both tool sets configured via `CHAT_TOOLS`).

### Debug Logging

Set `DEBUG=1` to enable detailed per-connection log files in `logs/`. Each connection (client,
chat, tts) writes a separate file with all sent and received messages. Useful for tracing
interruption handling and message flow.

## Client

### Python (desktop)

```bash
python -m client.client
```

Run from the `voice_note` directory. Install requirements from `client/requirements.txt` first (PyAudio requires PortAudio dev libraries). Hold the REC button to talk; releasing sends the audio for transcription and playback of the response starts as soon as audio arrives. Pressing REC again while the assistant is speaking interrupts it.

### Android

`android_app/` contains a Kotlin push-to-talk client (Ktor WebSockets, `AudioRecord`/`AudioTrack`, min SDK 31). Build it with Android Studio or:

```bash
cd android_app && ./gradlew assembleDebug
```

The APK is written to `app/build/outputs/apk/debug/app-debug.apk`. On first launch, grant microphone permission, then enter the server host and port in the app (default port `12345`) and press Save. Hold the record button to talk; releasing transcribes the utterance and streams the spoken response. Pressing the button while the assistant is speaking interrupts it (see [Interruption Mechanism](#interruption-mechanism)). There is also a reconnect button for when the server address or network changes.

## Tests

```bash
pip install pytest pytest-asyncio websockets
cd voice_note && pytest -q
```
