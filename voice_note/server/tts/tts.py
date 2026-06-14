from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import torch
from loguru import logger

# Make the local dots.tts clone importable without a formal install.
DOTS_TTS_ROOT = Path(__file__).resolve().parent / "dots.tts"
DOTS_TTS_SRC = DOTS_TTS_ROOT / "src"
for _path in (DOTS_TTS_ROOT, DOTS_TTS_SRC):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


def _install_optional_dep_stubs() -> None:
    """Stub out optional text-normalization / language-detection deps.

    dots.tts imports these unconditionally in ``utils.text``, but they are only
    used when ``normalize_text=True`` or a language tag is requested. The TTS
    server passes raw text through, so no-ops are sufficient.
    """
    if "tn" not in sys.modules:
        tn = ModuleType("tn")
        tn_english = ModuleType("tn.english")
        tn_english_normalizer = ModuleType("tn.english.normalizer")
        tn_chinese = ModuleType("tn.chinese")
        tn_chinese_normalizer = ModuleType("tn.chinese.normalizer")

        class _NopNormalizer:
            def normalize(self, text: str) -> str:
                return text

        tn_english_normalizer.Normalizer = _NopNormalizer
        tn_english.normalizer = tn_english_normalizer
        tn.english = tn_english
        tn_chinese_normalizer.Normalizer = _NopNormalizer
        tn_chinese.normalizer = tn_chinese_normalizer
        tn.chinese = tn_chinese
        for _name, _mod in (
            ("tn", tn),
            ("tn.english", tn_english),
            ("tn.english.normalizer", tn_english_normalizer),
            ("tn.chinese", tn_chinese),
            ("tn.chinese.normalizer", tn_chinese_normalizer),
        ):
            sys.modules[_name] = _mod

    if "langcodes" not in sys.modules:
        langcodes = ModuleType("langcodes")

        class _Language:
            language = "EN"

            @classmethod
            def get(cls, _code: str) -> Any:
                return cls()

            @classmethod
            def find(cls, _code: str) -> Any:
                return cls()

            def prefer_macrolanguage(self) -> Any:
                return self

        langcodes.Language = _Language
        sys.modules["langcodes"] = langcodes

    if "lingua" not in sys.modules:
        lingua = ModuleType("lingua")

        class _Lng:
            @classmethod
            def all(cls) -> list[Any]:
                return []

        class _Detector:
            def detect_language_of(self, _text: str) -> Any:
                return None

        class _Builder:
            @classmethod
            def from_languages(cls, *_languages: Any) -> Any:
                return cls()

            def build(self) -> Any:
                return _Detector()

        lingua.Language = _Lng
        lingua.LanguageDetectorBuilder = _Builder
        sys.modules["lingua"] = lingua


_install_optional_dep_stubs()

from dots_tts.runtime_double_streaming import (  # noqa: E402
    DotsTtsRuntimeDoubleStreaming,
    DoubleStreamingSession,
)

from server.base_server import BaseServer
from server.utils.streaming_connection import POLL_INTERVAL, StreamReset


DEVICE = "cuda"
assert torch.cuda.is_available(), "dots.tts TTS server requires CUDA"

MODEL_PATH = "/home/jonas/playground/s2t/voice_note/models/dots.tts-mf"
VOICE_PATH = Path(__file__).resolve().parent / "sample.wav"

PRECISION = "bfloat16"
NUM_STEPS = 4
GUIDANCE_SCALE = 1.2
MAX_GENERATE_LENGTH = 500
OPTIMIZE = True

WARMUP_TEXT = "Warmup."


class AsyncDotsTTSGenerator:
    """Async wrapper around a dots.tts double-streaming session.

    The runtime is loaded once and reused for every utterance. A new
    ``DoubleStreamingSession`` is created for each utterance and discarded on
    finish or interrupt.
    """

    def __init__(self, runtime: DotsTtsRuntimeDoubleStreaming, voice_path: str):
        self.runtime = runtime
        self.voice_path = voice_path
        self._text_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._audio_queue: asyncio.Queue[Optional[torch.Tensor]] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._started = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Currently active session, if any. ``interrupt()`` flips its
        # ``end_flag`` to make the decode loop bail out promptly before
        # cancelling the task. Cleared by the producer in a ``finally``.
        self._current_session: Optional[DoubleStreamingSession] = None

    def _ensure_started(self) -> None:
        if not self._started:
            self._loop = asyncio.get_running_loop()
            self._task = asyncio.create_task(self._run())
            self._started = True

    async def push_text(self, text: str) -> None:
        self._ensure_started()
        await self._text_queue.put(text)

    async def finish(self) -> None:
        self._ensure_started()
        await self._text_queue.put(None)

    async def get_audio_chunk(self) -> Optional[torch.Tensor]:
        self._ensure_started()
        return await self._audio_queue.get()

    def audio_available(self) -> bool:
        return not self._audio_queue.empty()

    async def interrupt(self) -> None:
        """Stop the in-flight session and restart generation from scratch."""
        # Tell any running session to bail out of its decode loop as soon as the
        # current chunk finishes. ``finish_text()`` re-checks ``end_flag`` at
        # the top of every iteration, so this avoids keeping the GPU busy on
        # audio that will only be discarded. The flag is a private field of
        # the model state, but the session exposes no public abort method.
        session = self._current_session
        if session is not None and not session.is_finished:
            try:
                session._state.end_flag = True
            except Exception:
                pass

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        self._current_session = None

        # Drain remaining queues.
        for queue in (self._text_queue, self._audio_queue):
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        self._text_queue = asyncio.Queue()
        self._audio_queue = asyncio.Queue()
        self._task = asyncio.create_task(self._run())
        self._started = True

    async def warmup(self) -> None:
        """Run a short throwaway synthesis so that CUDA caches/paths are warm."""
        self._ensure_started()
        await self._text_queue.put(WARMUP_TEXT)
        await self._text_queue.put(None)
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                break

    async def _drain_finish(
        self,
        session: DoubleStreamingSession,
        audio_queue: asyncio.Queue[Optional[torch.Tensor]],
    ) -> None:
        """Stream tail audio from ``session.finish_text()`` into ``audio_queue``.

        Each ``next()`` is dispatched to a worker thread via ``asyncio.to_thread``
        so cancellation is observed between chunks.
        """
        gen = session.finish_text()
        sentinel = object()
        try:
            while True:
                chunk = await asyncio.to_thread(next, gen, sentinel)
                if chunk is sentinel:
                    break
                await audio_queue.put(chunk.detach().cpu())
            await audio_queue.put(None)
        except asyncio.CancelledError:
            # Don't push a ``None`` while being cancelled; ``interrupt()`` will
            # drain and discard this queue anyway.
            raise
        except Exception:
            logger.exception("finish_text failed")
            await audio_queue.put(None)

    async def _run(self) -> None:
        # Capture the current audio queue locally. ``interrupt()`` swaps
        # ``self._audio_queue`` for a fresh queue when starting a new session,
        # so any chunk this task produces *after* that swap must land in the
        # *old* queue - which ``interrupt()`` then drains and discards. Using
        # ``self._audio_queue`` directly here would let a chunk from the
        # cancelled session slip into the new stream and be played with the
        # new request id.
        audio_queue = self._audio_queue
        try:
            while True:
                first_text = await self._text_queue.get()
                if first_text is None:
                    # Empty request: just signal end and wait for the next one.
                    await audio_queue.put(None)
                    continue

                session = self.runtime.start_double_streaming(
                    prompt_audio_path=self.voice_path,
                    ode_method="euler",
                    num_steps=NUM_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    eos_threshold=0.8,
                )
                self._current_session = session

                buffer = first_text
                token_cursor = 0
                token_ids: list[int] = []
                finished = False

                try:
                    # The model can decide the audio is complete (set
                    # ``end_flag``) before the consumer has pushed every text
                    # token. The ``end_flag`` part of the condition lets us
                    # bail out cooperatively; ``_drain_finish()`` then flushes
                    # the tail audio via ``finish_text()``, which has a
                    # dedicated "already at EOS" branch.
                    while (
                        not (finished and token_cursor >= len(token_ids))
                        and not session._state.end_flag
                    ):
                        # Pull in any newly arrived text.
                        got_new_text = False
                        while True:
                            try:
                                item = self._text_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                            if item is None:
                                finished = True
                            else:
                                buffer += item
                                got_new_text = True

                        # Re-tokenize only when the buffer changed.
                        if got_new_text:
                            token_ids = self.runtime.model.tokenizer.encode(
                                buffer.strip(), add_special_tokens=False
                            )

                        # Push all currently known tokens through the session.
                        while (
                            token_cursor < len(token_ids)
                            and not session._state.end_flag
                        ):
                            token_id = token_ids[token_cursor]
                            try:
                                chunk = await asyncio.to_thread(
                                    session.push_text_token, token_id
                                )
                            except RuntimeError:
                                # ``end_flag`` may have been flipped by a
                                # previous decode between the condition check
                                # above and the worker thread entering
                                # ``push_text_token``; treat that as a normal
                                # end. Anything else propagates.
                                if session._state.end_flag:
                                    break
                                raise
                            token_cursor += 1
                            if chunk is not None:
                                await audio_queue.put(chunk.detach().cpu())
                            # Yield briefly so cancellation / incoming text can be handled.
                            await asyncio.sleep(0)

                        if finished and token_cursor >= len(token_ids):
                            break

                        # Only sleep when waiting for more text or more tokens.
                        await asyncio.sleep(POLL_INTERVAL)

                    # Tail decoding runs cooperatively so cancellation can land
                    # between chunks; the per-chunk ``to_thread`` keeps the GPU
                    # busy without making the loop uncancellable.
                    await self._drain_finish(session, audio_queue)
                finally:
                    if self._current_session is session:
                        self._current_session = None
        except asyncio.CancelledError:
            logger.info("TTS generation task cancelled")
        except Exception:
            logger.exception("TTS generation task failed")
            await audio_queue.put(None)


class TTSServer(BaseServer):
    def __init__(self, host: str, port: int):
        super().__init__("tts", host, port)
        runtime = DotsTtsRuntimeDoubleStreaming.from_pretrained(
            MODEL_PATH,
            precision=PRECISION,
            optimize=OPTIMIZE,
            max_generate_length=MAX_GENERATE_LENGTH,
        )
        self.generator = AsyncDotsTTSGenerator(runtime, VOICE_PATH)
        self.audio_config = {
            "format": 1,  # pyaudio.paFloat32
            "channels": 1,
            "rate": runtime.sample_rate,
        }
        self._warmed_up = False

    async def serve_forever(self) -> None:
        # Warm up CUDA paths once at server start so the first client
        # connection doesn't pay for it. ``warmup()`` also starts the
        # generator's producer task and captures the event loop reference,
        # which has to happen on a running loop rather than from ``__init__``.
        if not self._warmed_up:
            logger.info("Running warmup synthesis...")
            await self.generator.warmup()
            self._warmed_up = True
            logger.info("Warmup complete.")
        await super().serve_forever()

    async def _handle_workload(self) -> None:
        current_id: Optional[str] = None
        received: list[dict[str, Any]] = []
        finished = False

        while True:
            try:
                received += self._recv_client_messages()

                # Discard stale data from a previous request id.
                if len(received) > 1 and received[0]["id"] != received[-1]["id"]:
                    received = [data for data in received if data["id"] == received[-1]["id"]]

                if len(received) > 0:
                    current_id = received[0]["id"]
                    text = "".join(msg["text"] for msg in received)
                    finished = received[-1]["status"] == "FINISHED"
                    await self.generator.push_text(text)
                    received = []
                    if finished:
                        await self.generator.finish()

                while current_id is not None and self.generator.audio_available():
                    audio = await self.generator.get_audio_chunk()
                    client = self.streams["client"]
                    if audio is None:
                        client.send(
                            {
                                "audio": b"",
                                "status": "FINISHED",
                                "id": current_id,
                                "config": self.audio_config,
                            }
                        )
                        # Do not interrupt the generator for a clean end-of-utterance;
                        # it will wait for the next text on its own.
                        current_id = None
                        finished = False
                    else:
                        bytes_ = audio.float().squeeze().cpu().numpy().tobytes()
                        client.send(
                            {
                                "audio": bytes_,
                                "status": "GENERATING",
                                "id": current_id,
                                "config": self.audio_config,
                            }
                        )

                await asyncio.sleep(POLL_INTERVAL)
            except StreamReset:
                await self.generator.interrupt()
                current_id = None
                received = []
                finished = False
            except ConnectionError:
                break


if __name__ == "__main__":
    asyncio.run(TTSServer("0.0.0.0", 12347).serve_forever())
