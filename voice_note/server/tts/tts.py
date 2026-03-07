import asyncio
import typing as tp
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from server.base_server import BaseServer
from server.utils.streaming_connection import POLL_INTERVAL, StreamReset

from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

MODEL_DIR = Path("./models/Qwen3-TTS-12Hz-0.6B-Base")
REF_AUDIO = Path("./sample.wav")
LANGUAGE = "German"

MAX_TOKENS = 2048
FIRST_STREAM_AUDIO_CHUNK_TOKENS = 12
STREAM_AUDIO_CHUNK_TOKENS = 24
TOKEN_HOLDBACK = 1
DO_SAMPLE = True
TOP_K = 50
TOP_P = 1.0
TEMPERATURE = 0.9
ATTN_MODE = "default"
DEFAULT_SAMPLE_RATE = 24000
FIRST_TEXT_CHUNK_MIN_CHARS = 24


def split_flushable_text(text: str, is_first_chunk: bool, final_chunk: bool) -> tuple[str, str]:
    """Return a boundary-safe prefix to flush and the remaining buffered suffix."""
    if final_chunk:
        return text.strip(), ""

    if not text:
        return "", ""

    sentence_boundary = max(text.rfind(mark) for mark in ".!?;:")
    if sentence_boundary >= 0:
        split_at = sentence_boundary + 1
        return text[:split_at].strip(), text[split_at:].lstrip()

    whitespace_boundary = max(text.rfind(" "), text.rfind("\n"), text.rfind("\t"))
    if whitespace_boundary < 0:
        return "", text

    split_at = whitespace_boundary + 1
    if is_first_chunk and split_at < FIRST_TEXT_CHUNK_MIN_CHARS:
        return "", text
    return text[:split_at].strip(), text[split_at:].lstrip()


class StreamingTextState:
    """Manage incremental text for coupled text/audio streaming.

    The model reads text and generates audio step-by-step at the same time.
    For each audio step, it expects the matching text representation to exist.

    Text arrives in chunks, so the newest tokens near the current chunk boundary
    are often not final yet (the next chunk can change tokenization around that
    boundary). This class keeps a small holdback and only releases text states
    once they are stable.

    It also returns only the newly stable part each time, so the caller can
    append it to the existing buffer and keep text/audio step alignment correct.
    """

    def __init__(self, tts_model: Qwen3TTSModel, holdback_tokens: int = 2):
        self.tts = tts_model
        self.talker = tts_model.model.talker
        self.config = tts_model.model.config
        self.holdback_tokens = holdback_tokens
        self.full_text = ""
        self.committed_trailing_ids: list[int] = []
        self.finalized = False
        self._empty_assistant_ids = self._assistant_ids("")[0].tolist()

    def _assistant_ids(self, text: str) -> torch.Tensor:
        """Tokenize text with the assistant prompt format used by Qwen TTS."""
        return self.tts._tokenize_texts([self.tts._build_assistant_text(text)])[0]

    @staticmethod
    def _common_prefix_len(a: list[int], b: list[int]) -> int:
        """Return the length of the common prefix shared by two token lists."""
        n = min(len(a), len(b))
        idx = 0
        while idx < n and a[idx] == b[idx]:
            idx += 1
        return idx

    @staticmethod
    def _common_suffix_len(a: list[int], b: list[int], start_a: int, start_b: int) -> int:
        """Return common suffix length while respecting protected prefix ranges."""
        i = len(a) - 1
        j = len(b) - 1
        count = 0
        while i >= start_a and j >= start_b and a[i] == b[j]:
            count += 1
            i -= 1
            j -= 1
        return count

    def _extract_content_ids(self, full_ids: list[int]) -> list[int]:
        """Strip assistant wrapper tokens and return only user text token IDs."""
        prefix = self._common_prefix_len(full_ids, self._empty_assistant_ids)
        suffix = self._common_suffix_len(full_ids, self._empty_assistant_ids, prefix, prefix)
        end = len(full_ids) - suffix if suffix > 0 else len(full_ids)
        if end < prefix:
            return []
        content = full_ids[prefix:end]
        if content:
            # The first content token is already consumed in talker prefill.
            content = content[1:]
        return content

    def push_text(self, chunk: str, final_chunk: bool) -> torch.Tensor | None:
        """Append incoming text and return only *new* committed trailing hidden states.

        The returned tensor contains hidden vectors for the token IDs that became
        stable since the previous call. It is not the full trailing-text history.
        The caller appends these new vectors to its existing trailing buffer, so
        decoder-step index and trailing-text index stay aligned over time.

        Returns None when no additional stable tokens are available yet.
        """
        if self.finalized:
            return None

        # Keep tokenization stable across chunk boundaries by inserting spaces when needed.
        if self.full_text and chunk and not chunk.startswith((" ", ".", ",", "!", "?", ";", ":")):
            self.full_text += " "
        self.full_text += chunk
        all_ids = self._assistant_ids(self.full_text)[0].tolist()
        candidate_ids = self._extract_content_ids(all_ids)

        if final_chunk:
            stable_ids = candidate_ids + [self.config.tts_eos_token_id]
            self.finalized = True
        elif len(candidate_ids) <= self.holdback_tokens:
            stable_ids = []
        else:
            stable_ids = candidate_ids[:-self.holdback_tokens]

        common_prefix = self._common_prefix_len(self.committed_trailing_ids, stable_ids)
        if common_prefix < len(self.committed_trailing_ids):
            stable_ids = self.committed_trailing_ids

        new_ids = stable_ids[len(self.committed_trailing_ids):]
        if not new_ids:
            return None

        self.committed_trailing_ids.extend(new_ids)
        new_ids_tensor = torch.tensor([new_ids], device=self.talker.device, dtype=torch.long)
        return self.talker.text_projection(self.talker.get_text_embeddings()(new_ids_tensor))

    def current_input_ids(self) -> torch.Tensor:
        """Return tokenized full text for talker prefill."""
        return self._assistant_ids(self.full_text)

    def trailing_text_hidden(self) -> torch.Tensor:
        """Return projected hidden states for all currently committed trailing text IDs."""
        if not self.committed_trailing_ids:
            hidden_size = self.talker.config.hidden_size
            return torch.empty((1, 0, hidden_size), device=self.talker.device, dtype=self.talker.dtype)
        ids_tensor = torch.tensor([self.committed_trailing_ids], device=self.talker.device, dtype=torch.long)
        return self.talker.text_projection(self.talker.get_text_embeddings()(ids_tensor))


def build_talker_inputs_xvector(
    tts_model: Qwen3TTSModel,
    input_id: torch.Tensor,
    voice_clone_prompt: tp.Any,
    language: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create initial talker embeddings for x-vector voice cloning prefill."""
    talker = tts_model.model.talker
    cfg = tts_model.model.config

    voice_clone_spk_embeds = tts_model.model.generate_speaker_prompt(voice_clone_prompt)
    speaker_embed = voice_clone_spk_embeds[0]

    language_lower = language.lower()
    if language_lower == "auto":
        language_id = None
    else:
        language_id = cfg.talker_config.codec_language_id[language_lower]

    tts_bos_embed, _, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(
            torch.tensor(
                [[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]],
                device=talker.device,
                dtype=input_id.dtype,
            )
        )
    ).chunk(3, dim=1)

    if language_id is None:
        codec_prefill_list = [[
            cfg.talker_config.codec_nothink_id,
            cfg.talker_config.codec_think_bos_id,
            cfg.talker_config.codec_think_eos_id,
        ]]
    else:
        codec_prefill_list = [[
            cfg.talker_config.codec_think_id,
            cfg.talker_config.codec_think_bos_id,
            language_id,
            cfg.talker_config.codec_think_eos_id,
        ]]

    codec_input_embedding_0 = talker.get_input_embeddings()(
        torch.tensor(codec_prefill_list, device=talker.device, dtype=input_id.dtype)
    )
    codec_input_embedding_1 = talker.get_input_embeddings()(
        torch.tensor(
            [[cfg.talker_config.codec_pad_id, cfg.talker_config.codec_bos_id]],
            device=talker.device,
            dtype=input_id.dtype,
        )
    )
    codec_input_embedding = torch.cat(
        [codec_input_embedding_0, speaker_embed.view(1, 1, -1), codec_input_embedding_1],
        dim=1,
    )

    talker_input_embed_role = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
    talker_input_embed = torch.cat(
        (tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1), tts_bos_embed),
        dim=1,
    ) + codec_input_embedding[:, :-1]
    talker_input_embed = torch.cat((talker_input_embed_role, talker_input_embed), dim=1)

    first_text_token = input_id[:, 3:4]
    if first_text_token.shape[1] != 1:
        raise ValueError("Need at least one text token in the initial stream chunk.")
    talker_input_embed = torch.cat(
        [
            talker_input_embed,
            talker.text_projection(talker.get_text_embeddings()(first_text_token)) + codec_input_embedding[:, -1:],
        ],
        dim=1,
    )

    attention_mask = torch.ones((1, talker_input_embed.shape[1]), device=talker.device, dtype=torch.long)
    return talker_input_embed, attention_mask, tts_pad_embed


def sample_next_token(logits: torch.Tensor) -> torch.Tensor:
    """Sample next codec token with temperature/top-k/top-p controls."""
    if not DO_SAMPLE:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scores = logits / max(TEMPERATURE, 1e-5)

    if TOP_K > 0:
        topk_vals, topk_idx = torch.topk(scores, k=min(TOP_K, scores.shape[-1]), dim=-1)
        masked = torch.full_like(scores, -float("inf"))
        masked.scatter_(1, topk_idx, topk_vals)
        scores = masked

    if TOP_P < 1.0:
        sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > TOP_P
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, -float("inf"))
        scores = torch.full_like(scores, -float("inf"))
        scores.scatter_(1, sorted_idx, sorted_scores)

    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class AsyncTTSGenerator:
    def __init__(self):
        """Initialize model state and runtime queues for one streaming session."""
        self.text_queue: asyncio.Queue[str] = asyncio.Queue()
        self.audio_queue: asyncio.Queue[torch.Tensor | None] = asyncio.Queue()
        self.generation_task: asyncio.Task | None = None
        self.finished = False

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for TTS server.")
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tts, self.config = self._build_tts()

        self.codec_eos_id = self.config.talker_config.codec_eos_token_id
        self.codebook_size = int(self.tts.model.speech_tokenizer.model.config.decoder_config.codebook_size)
        self.suppress_tokens = [
            i for i in range(self.codebook_size, self.config.talker_config.vocab_size) if i != self.codec_eos_id
        ]

        # Build the voice-clone prompt once at startup.
        # In practice this path is stable when tokenizer encode runs on CPU.
        self._move_speech_tokenizer_to_cpu()
        prompt_items = self.tts.create_voice_clone_prompt(str(REF_AUDIO), x_vector_only_mode=True)
        self.voice_clone_prompt = self.tts._prompt_items_to_voice_clone_prompt(prompt_items)

        # Decode is much faster on CUDA, so move tokenizer back after prompt construction.
        self._move_speech_tokenizer_to_device(self.device)
        self.sample_rate = DEFAULT_SAMPLE_RATE
        print(f"[tts] loaded on {self.device} ({self.dtype})")

    def _move_speech_tokenizer_to_device(self, device: torch.device) -> None:
        """Move tokenizer wrapper + underlying model and keep wrapper device in sync."""
        speech_tokenizer = getattr(self.tts.model, "speech_tokenizer", None)
        if speech_tokenizer is None:
            return
        if hasattr(speech_tokenizer, "device"):
            speech_tokenizer.device = device
        if hasattr(speech_tokenizer, "to"):
            try:
                speech_tokenizer.to(device)
            except Exception as exc:
                print(f"[tts] tokenizer wrapper move to {device} failed: {exc}")
        submodel = getattr(speech_tokenizer, "model", None)
        if submodel is None or not hasattr(submodel, "to"):
            return
        try:
            submodel.to(device)
        except Exception as exc:
            print(f"[tts] tokenizer move to {device} failed: {exc}")
            return
        if hasattr(submodel, "eval"):
            submodel.eval()

    def _move_speech_tokenizer_to_cpu(self) -> None:
        """Convenience helper for tokenizer CPU placement."""
        self._move_speech_tokenizer_to_device(torch.device("cpu"))

    def _build_tts(self) -> tuple[Qwen3TTSModel, tp.Any]:
        """Load Qwen TTS talker model and processor from local model files."""
        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"Qwen model directory not found: {MODEL_DIR}")
        if not REF_AUDIO.exists():
            raise FileNotFoundError(f"Reference audio not found: {REF_AUDIO}")

        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        base_load_kwargs = {
            "local_files_only": True,
            "use_safetensors": True,
            "dtype": self.dtype,
        }

        if ATTN_MODE == "auto":
            attn_candidates = ["flash_attention_2", None]
        elif ATTN_MODE in {"none", "", "default"}:
            attn_candidates = [None]
        else:
            attn_candidates = [ATTN_MODE]

        model = None
        selected_attn = None
        last_err: Exception | None = None
        for attn_impl in attn_candidates:
            load_kwargs = dict(base_load_kwargs)
            if attn_impl is not None:
                load_kwargs["attn_implementation"] = attn_impl
            try:
                model = AutoModel.from_pretrained(MODEL_DIR, **load_kwargs)
                selected_attn = attn_impl
                break
            except Exception as exc:
                last_err = exc

        if model is None:
            raise RuntimeError(f"Failed to load model with attention mode(s): {attn_candidates}") from last_err

        model = model.to(self.device)
        model.eval()
        print(f"Loaded Qwen3 TTS on {self.device.type} ({self.dtype}, attn={selected_attn or 'default'})")

        processor = AutoProcessor.from_pretrained(
            MODEL_DIR,
            fix_mistral_regex=True,
            local_files_only=True,
        )
        tts = Qwen3TTSModel(model, processor, model.generate_config)
        return tts, tts.model.config

    async def start(self):
        """Start background generation loop if it is not already running."""
        if self.generation_task and not self.generation_task.done():
            return
        self.generation_task = asyncio.create_task(self._generation_loop())

    async def add_text(self, text: str):
        """Queue a text chunk for incremental synthesis."""
        normalized = text.strip()
        if not normalized:
            return
        print(f"---> Streaming in text: '{normalized}'")
        await self.text_queue.put(normalized)

    async def finish(self):
        """Mark the current request as complete (no more text chunks expected)."""
        self.finished = True

    async def get_audio_chunk(self) -> torch.Tensor | None:
        """Get next produced PCM chunk, or None as end-of-stream sentinel."""
        return await self.audio_queue.get()

    async def restart(self):
        """Cancel active generation and reset queues for a new request."""
        print("[tts] restarting generator")
        if self.generation_task:
            self.generation_task.cancel()
            try:
                await self.generation_task
            except asyncio.CancelledError:
                pass
        self.generation_task = None
        self.text_queue = asyncio.Queue()
        self.audio_queue = asyncio.Queue()
        self.finished = False
        await self.start()

    async def _generation_loop(self):
        """Run one full-duplex generation session for queued text chunks."""
        try:
            while self.text_queue.empty():
                if self.finished:
                    raise RuntimeError("Generation finished before any text chunk was received.")
                await asyncio.sleep(POLL_INTERVAL)
            first_chunk = self.text_queue.get_nowait()
            streamer = StreamingTextState(self.tts, holdback_tokens=TOKEN_HOLDBACK)

            first_chunk_is_final = self.finished and self.text_queue.empty()
            streamer.push_text(first_chunk, final_chunk=first_chunk_is_final)

            input_ids = streamer.current_input_ids()
            talker_input_embeds, attention_mask, tts_pad_embed = build_talker_inputs_xvector(
                tts_model=self.tts,
                input_id=input_ids,
                voice_clone_prompt=self.voice_clone_prompt,
                language=LANGUAGE,
            )
            trailing_text_hidden = streamer.trailing_text_hidden()

            prefill = self._talker_call(
                inputs_embeds=talker_input_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
            )

            past_key_values = prefill.past_key_values
            past_hidden = prefill.past_hidden
            generation_step = prefill.generation_step
            input_token = torch.tensor(
                [[self.config.talker_config.codec_bos_id]],
                device=self.tts.model.device,
                dtype=attention_mask.dtype,
            )

            prefill_len = int(attention_mask.shape[1])
            full_seq_len = prefill_len + MAX_TOKENS
            attention_mask_full = torch.zeros((1, full_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask_full[:, :prefill_len] = 1
            cache_position = torch.empty((1,), device=attention_mask.device, dtype=torch.long)

            codec_buffer: list[torch.Tensor] = []
            emitted_chunk_count = 0
            finalized_text = first_chunk_is_final

            for _ in range(MAX_TOKENS):
                # Gate decoder steps until matching trailing text hidden states are available.
                while generation_step >= trailing_text_hidden.shape[1] and (not self.finished or not self.text_queue.empty()):
                    pushed = False
                    while not self.text_queue.empty():
                        chunk = self.text_queue.get_nowait()
                        is_final = self.finished and self.text_queue.empty()
                        new_hidden = streamer.push_text(chunk, final_chunk=is_final)
                        if new_hidden is not None:
                            trailing_text_hidden = torch.cat([trailing_text_hidden, new_hidden], dim=1)
                        pushed = True
                        finalized_text = finalized_text or is_final

                    if not pushed and self.finished and not finalized_text:
                        new_hidden = streamer.push_text("", final_chunk=True)
                        if new_hidden is not None:
                            trailing_text_hidden = torch.cat([trailing_text_hidden, new_hidden], dim=1)
                        finalized_text = True
                        pushed = True

                    if pushed:
                        continue
                    await asyncio.sleep(POLL_INTERVAL)

                past_len = prefill_len + generation_step
                attention_mask_full[:, past_len] = 1
                step_attention_mask = attention_mask_full[:, : past_len + 1]
                cache_position.fill_(past_len)

                outputs = self._talker_call(
                    input_ids=input_token,
                    attention_mask=step_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=False,
                    return_dict=True,
                    past_hidden=past_hidden,
                    generation_step=generation_step,
                    trailing_text_hidden=trailing_text_hidden,
                    tts_pad_embed=tts_pad_embed,
                    cache_position=cache_position,
                )

                logits = outputs.logits[:, -1, :]
                if self.suppress_tokens:
                    logits = logits.clone()
                    logits[:, self.suppress_tokens] = -float("inf")
                input_token = sample_next_token(logits)

                if outputs.hidden_states is None or outputs.hidden_states[1] is None:
                    raise RuntimeError("Missing codec ids in talker output.")
                codec_ids = outputs.hidden_states[1].squeeze(0)
                if (codec_ids < 0).any() or (codec_ids >= self.codebook_size).any():
                    codec_ids = codec_ids.clamp(0, self.codebook_size - 1)
                codec_buffer.append(codec_ids)

                past_key_values = outputs.past_key_values
                past_hidden = outputs.past_hidden
                generation_step = outputs.generation_step

                if input_token.item() == self.codec_eos_id:
                    break

                chunk_target_tokens = (
                    FIRST_STREAM_AUDIO_CHUNK_TOKENS if emitted_chunk_count == 0 else STREAM_AUDIO_CHUNK_TOKENS
                )
                if len(codec_buffer) >= chunk_target_tokens:
                    # Emit the first chunk earlier, then use larger chunks for better throughput.
                    await self._decode_and_emit(codec_buffer)
                    codec_buffer = []
                    emitted_chunk_count += 1
                    await asyncio.sleep(0)

            if codec_buffer:
                await self._decode_and_emit(codec_buffer)
        except asyncio.CancelledError:
            print("[tts] generation loop cancelled")
            raise
        finally:
            await self.audio_queue.put(None)

    async def _decode_and_emit(self, codec_buffer: list[torch.Tensor]) -> None:
        """Decode accumulated codec tokens to PCM and enqueue them for streaming."""
        codes_tensor = torch.stack(codec_buffer, dim=0).to(self.device)
        with torch.inference_mode():
            wavs, sample_rate = self.tts.model.speech_tokenizer.decode([{"audio_codes": codes_tensor}])
        self.sample_rate = int(sample_rate)
        audio_np = np.asarray(wavs[0], dtype=np.float32)
        audio_torch = torch.from_numpy(audio_np)
        await self.audio_queue.put(audio_torch)
        await asyncio.sleep(0)

    def _talker_call(self, **kwargs):
        """Call talker model in inference mode."""
        with torch.inference_mode():
            return self.tts.model.talker(**kwargs)


class TTSServer(BaseServer):
    def __init__(self, host: str, port: int):
        """Create TTS server and initialize model-backed generator."""
        super().__init__("tts", host, port)
        self.generator = AsyncTTSGenerator()

        self.audio_config = {
            "format": 1,
            "channels": 1,
        }

    async def _handle_workload(self) -> None:
        """Bridge client text streaming to generator and stream audio chunks back."""
        await self.generator.start()

        current_id = None
        received = []
        pending_text = ""
        finished = False
        first_text_chunk_sent = False
        while True:
            try:
                received += self._recv_client_messages()

                if len(received) > 1 and received[0]["id"] != received[-1]["id"]:
                    received = [data for data in received if data["id"] == received[-1]["id"]]
                    pending_text = ""
                    first_text_chunk_sent = False

                if len(received) > 0:
                    current_id = received[0]["id"]
                    pending_text += "".join(msg["text"] for msg in received)
                    finished = received[-1]["status"] == "FINISHED"

                    flush_text, pending_text = split_flushable_text(
                        pending_text,
                        is_first_chunk=not first_text_chunk_sent,
                        final_chunk=finished,
                    )
                    if flush_text:
                        await self.generator.add_text(flush_text)
                        first_text_chunk_sent = True
                    received = []

                    if finished or flush_text == "Let me think about that.":
                        await self.generator.finish()

                if current_id is not None and not self.generator.audio_queue.empty():
                    while current_id is not None and not self.generator.audio_queue.empty():
                        audio = await self.generator.get_audio_chunk()
                        if audio is None:
                            if finished:
                                self.streams["client"].send({"audio": b"", "status": "FINISHED", "id": current_id})
                            await self.generator.restart()
                            current_id = None
                            pending_text = ""
                            finished = False
                            first_text_chunk_sent = False
                        else:
                            bytes_ = audio.cpu().numpy().tobytes()
                            self.audio_config["rate"] = self.generator.sample_rate
                            self.streams["client"].send(
                                {
                                    "audio": bytes_,
                                    "status": "GENERATING",
                                    "id": current_id,
                                    "config": self.audio_config,
                                }
                            )

                await asyncio.sleep(POLL_INTERVAL)
            except StreamReset:
                await self.generator.restart()
                current_id = None
                pending_text = ""
                finished = False
                first_text_chunk_sent = False
            except ConnectionError:
                break


if __name__ == "__main__":
    asyncio.run(TTSServer("0.0.0.0", 12347).serve_forever())
