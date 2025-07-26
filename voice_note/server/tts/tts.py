import asyncio
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import ConditionAttributes, Entry, LMGen, TTSModel
from pathlib import Path
import torch
import typing as tp

from server.base_server import BaseServer, POLL_INTERVAL
from server.utils.streaming_connection import RESET_SENTINEL


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_DIR = Path('./models/tts-1.6b-en_fr')
VOICE_PATH = Path('./models/tts-voices/expresso/ex03-ex01_happy_001_channel1_334s.wav.1e68beda@240.safetensors')


class AsyncTTSGenerator:
    """
    A class to handle TTS generation asynchronously.

    It manages the model state and runs the generation loop in a background task,
    allowing for streaming of text input and audio output.
    """

    def __init__(self, tts_model: TTSModel, condition_attributes: ConditionAttributes):
        self.tts_model = tts_model
        self.condition_attributes = condition_attributes
        self.device = tts_model.lm.device
        self.generation_task: tp.Optional[asyncio.Task] = None
        self._reset_state()

    def _reset_state(self):
        """Resets the generator's internal state."""
        self.text_queue: asyncio.Queue[tp.Optional[Entry]] = asyncio.Queue()
        self.audio_queue: asyncio.Queue[tp.Optional[torch.Tensor]] = asyncio.Queue()
        self.lm_gen: tp.Optional[LMGen] = None
        self.state: tp.Optional[tp.Any] = None
        self.offset = 0
        self.finished = False
        self.first_chunk_processed = False

    def _on_text_hook(self, text_tokens: torch.Tensor):
        """Hook to inject our text tokens into the generation process."""
        predicted_token = text_tokens.item()
        assert self.state is not None

        if not self.state.entries and not self.state.queued and not self.finished:
            if predicted_token == self.tts_model.machine.token_ids.new_word:
                predicted_token = self.tts_model.machine.token_ids.pad

        output_token, consumed_new_word = self.tts_model.machine.process(self.offset, self.state, predicted_token)
        if consumed_new_word:
            word, step = self.state.transcript[-1]
            print(f"Step {step}: Model consumed word -> '{word}'")
        text_tokens[0] = output_token

    def _on_audio_hook(self, audio_tokens: torch.Tensor):
        """Hook to handle initial audio delay."""
        lm = self.tts_model.lm
        machine = self.tts_model.machine
        audio_offset = lm.audio_offset
        delays = lm.delays
        for q in range(audio_tokens.shape[1]):
            delay = delays[q + audio_offset]
            if self.offset < delay + self.tts_model.delay_steps:
                audio_tokens[:, q] = machine.token_ids.zero

    def _create_lm_gen(self) -> LMGen:
        """Creates and configures the LMGen instance."""
        assert self.tts_model.lm.condition_provider is not None
        prepared = self.tts_model.lm.condition_provider.prepare([self.condition_attributes])
        condition_tensors = self.tts_model.lm.condition_provider(prepared)

        return LMGen(
            self.tts_model.lm,
            temp=self.tts_model.temp,
            temp_text=self.tts_model.temp,
            cfg_coef=self.tts_model.cfg_coef,
            condition_tensors=condition_tensors,
            on_text_hook=self._on_text_hook,
            on_audio_hook=self._on_audio_hook,
        )

    async def _generation_loop(self):
        """The main background task for generating audio."""
        self.lm_gen = self._create_lm_gen()
        self.state = self.tts_model.machine.new_state([])
        self.offset = 0
        mimi = self.tts_model.mimi
        lm = self.tts_model.lm
        machine = self.tts_model.machine

        try:
            with self.lm_gen.streaming(batch_size=1), mimi.streaming(batch_size=1):
                while True:
                    if self.text_queue.empty() and not self.state.entries and not self.state.queued and not self.finished:
                        await asyncio.sleep(POLL_INTERVAL)
                        continue

                    while not self.text_queue.empty():
                        entry = self.text_queue.get_nowait()
                        if entry is None:
                            self.text_queue.put_nowait(None)
                            break
                        assert self.state is not None
                        self.state.entries.append(entry)

                    no_pending_entries = not self.state.entries and not self.state.queued
                    end_signaled = self.state.end_step is not None

                    if self.finished and no_pending_entries and end_signaled:
                        if self.offset >= self.state.end_step + self.tts_model.delay_steps + self.tts_model.final_padding:
                            break

                    with torch.no_grad():
                        missing_streams = lm.n_q - lm.dep_q
                        input_tokens = torch.full((1, missing_streams, 1), machine.token_ids.zero, dtype=torch.long,
                                                  device=self.device)
                        frame = self.lm_gen.step(input_tokens)

                        if frame is not None:
                            audio_codes = frame[:, 1:, :]
                            if (audio_codes < 0).any():
                                pcm = torch.zeros(mimi.frame_size, device=self.device)
                            else:
                                pcm = mimi.decode(audio_codes)
                                pcm = torch.clip(pcm.squeeze(0).squeeze(0), -1, 1)
                            await self.audio_queue.put(pcm)

                    self.offset += 1
                    await asyncio.sleep(POLL_INTERVAL)
        except asyncio.CancelledError:
            print("Generation loop cancelled.")
        finally:
            await self.audio_queue.put(None)

    async def start(self):
        """Starts the generation background task."""
        if self.generation_task and not self.generation_task.done():
            return
        self.generation_task = asyncio.create_task(self._generation_loop())

    async def add_text(self, text: str):
        """Adds a piece of text to be synthesized."""
        print(f"---> Streaming in text: '{text}'")
        entries = self.tts_model.prepare_script([text], padding_between=1)

        if not self.first_chunk_processed:
            if entries:
                self.first_chunk_processed = True
        else:
            if entries and entries[0].tokens:
                token_ids = self.tts_model.machine.token_ids
                if entries[0].tokens[0] in [token_ids.main, token_ids.other]:
                    entries[0].tokens.pop(0)

        for entry in entries:
            await self.text_queue.put(entry)

    async def finish(self):
        """Signals that no more text will be added."""
        self.finished = True
        await self.text_queue.put(None)

    async def get_audio_chunk(self) -> tp.Optional[torch.Tensor]:
        """Retrieves the next available chunk of audio."""
        return await self.audio_queue.get()

    async def restart(self):
        """Stops the current generation and resets the state for a new one."""
        print("\nRestarting generator...")
        if self.generation_task:
            self.generation_task.cancel()
            try:
                await self.generation_task
            except asyncio.CancelledError:
                pass
        
        self._reset_state()
        await self.start()


class TTSServer(BaseServer):
    def __init__(self, host: str, port: int):
        super().__init__("tts", host, port)
        checkpoint_info = CheckpointInfo.from_hf_repo(
            "DUMMY_REPO",
            moshi_weights=MODEL_DIR / "dsm_tts_1e68beda@240.safetensors",
            mimi_weights=MODEL_DIR / "tokenizer-e351c8d8-checkpoint125.safetensors",
            tokenizer=MODEL_DIR / "tokenizer_spm_8k_en_fr_audio.model",
            config_path=MODEL_DIR / "config.json",
        )
        tts_model = TTSModel.from_checkpoint_info(checkpoint_info, n_q=32, temp=0.6, device=DEVICE)
        condition_attributes = tts_model.make_condition_attributes([VOICE_PATH], cfg_coef=2.0)
        self.generator = AsyncTTSGenerator(tts_model, condition_attributes)

        self.audio_config = {
            'format': 1,
            'channels': 1,
            'rate': tts_model.mimi.sample_rate
        }

    async def _main_loop(self) -> None:
        await self.generator.start()
        
        input_task = asyncio.create_task(self._handle_input())
        output_task = asyncio.create_task(self._handle_output())

        done, pending = await asyncio.wait(
            [input_task, output_task], 
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            if task.exception():
                print(f"Error in TTS main loop: {task.exception()}")
        
        for task in pending:
            task.cancel()

    async def _handle_input(self):
        """Handles incoming text from the client."""
        client_stream = self.streams['client']
        while True:
            try:
                msg = await client_stream.received_q.get()
                
                if msg is RESET_SENTINEL:
                    print("TTS input handler received reset sentinel.")
                    await self.generator.restart()
                    continue

                await self.generator.add_text(msg['text'])
                if msg['status'] == 'FINISHED':
                    await self.generator.finish()
                
                client_stream.received_q.task_done()

            except ConnectionError:
                print("Connection lost in TTS input handler.")
                break

    async def _handle_output(self):
        """Handles sending generated audio back to the client."""
        client_stream = self.streams['client']
        current_id = None
        while True:
            try:
                audio_chunk = await self.generator.get_audio_chunk()
                self.generator.audio_queue.task_done()

                if client_stream.communication_id:
                    current_id = client_stream.communication_id

                if audio_chunk is None:
                    if current_id:
                        client_stream.send({'audio': b'', 'status': 'FINISHED', 'id': current_id})
                    # Don't restart here, wait for the input loop to trigger it on a new message
                else:
                    if current_id:
                        bytes_ = audio_chunk.cpu().numpy().tobytes()
                        client_stream.send({
                            'audio': bytes_,
                            'status': 'GENERATING',
                            'id': current_id,
                            'config': self.audio_config
                        })

            except ConnectionError:
                print("Connection lost in TTS output handler.")
                break


if __name__ == '__main__':
    server = TTSServer('0.0.0.0', 12347)
    asyncio.run(server.serve_forever())
