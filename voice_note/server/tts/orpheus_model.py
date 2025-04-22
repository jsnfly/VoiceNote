from pathlib import Path
from snac import SNAC
import torch
from transformers import LlamaForCausalLM, AutoTokenizer

LLM_PATH = Path("./models/orpheus-3b-0.1-ft")
DECODER_PATH = Path("./models/snac_24khz")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class OrpheusModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
        self.llm = LlamaForCausalLM.from_pretrained(LLM_PATH, use_safetensors=True, local_files_only=True,
                                                    torch_dtype=torch.bfloat16).to(DEVICE)
        self.decoder = SNAC.from_config(DECODER_PATH / "config.json")
        self.decoder.load_state_dict(torch.load(DECODER_PATH / "pytorch_model.bin"))
        self.decoder.eval()
        self.decoder.to(DEVICE, dtype=torch.bfloat16)
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

    def convert_to_audio(self, token_ids):
        with torch.inference_mode():
            token_ids = token_ids[:(len(token_ids) // 7)*7]

            indices_0 = range(0, len(token_ids), 7)
            indices_1 = torch.tensor([
                range(1, len(token_ids), 7),
                range(4, len(token_ids), 7)
            ]).T.flatten()
            indices_2 = torch.tensor([
                range(2, len(token_ids), 7),
                range(3, len(token_ids), 7),
                range(5, len(token_ids), 7),
                range(6, len(token_ids), 7)
            ]).T.flatten()

            codes = [token_ids[indices_0], token_ids[indices_1], token_ids[indices_2]]
            audio_hat = self.decoder.decode([c.unsqueeze(0) for c in codes])
            return audio_hat.squeeze().float()
