import json
import whisper
from pathlib import Path
from socket import create_server
from utils.audio import AudioConfig
from utils.sample import Sample
from utils.message import recv_message, send_message, recv_bytes_stream
from transformers import AutoTokenizer, LlamaForCausalLM

PORT = 12345
WHISPER_MODEL = 'medium'
LLAMA_MODEL = 'lmsys/vicuna-13b-v1.5'  # `Ǹone` to not use chat.
SAVE_DIR = Path(__file__).parent.resolve() / 'outputs'


def load_llama():
    if LLAMA_MODEL is None:
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    model = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL,
        load_in_4bit=True,
        device_map='auto',
    )
    return tokenizer, model


def handle_deletion(save_path):
    save_path = Path(save_path)
    for file in save_path.iterdir():
        file.unlink()
    save_path.rmdir()
    print(f"Deleted {save_path}.")


def transcribe(bytes_, audio_config, topic):
    sample = Sample([bytes_], audio_config)
    sample.transcribe(whisper_model, whisper_options)
    save_path = sample.save(SAVE_DIR / topic)
    print(sample.result.text)
    return sample.result.text, save_path


def add_to_metadata(save_path, data):
    metadata_path = Path(save_path) / 'metadata.json'
    if metadata_path.exists():
        with metadata_path.open() as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata = metadata | data
    with metadata_path.open('w') as f:
        json.dump(metadata, f, sort_keys=True, indent=4, ensure_ascii=False)


def get_chat_response(query):
    # TODO: cleanup, flash_attention2

    if llama_model is None:
        return 'Chat not available.'
    prompt = f"USER: {query}\nASSISTANT:"
    generation_config = llama_model.generation_config
    generation_config.max_length = len(llama_tokenizer(prompt)['input_ids']) + 512
    generation_config.top_k = 1
    generation_config.do_sample = True
    generation_config.num_return_sequences = 1
    generation_config.temperature = 1.5

    token_ids = llama_model.generate(
        llama_tokenizer(prompt, return_tensors='pt')['input_ids'].cuda(),
        generation_config
    )
    return llama_tokenizer.decode(token_ids[0], skip_special_tokens=True)[len(prompt):]


if __name__ == '__main__':
    whisper_model = whisper.load_model(WHISPER_MODEL, device='cuda')
    whisper_options = whisper.DecodingOptions()

    llama_tokenizer, llama_model = load_llama()
    with create_server(('0.0.0.0', PORT)) as sock:
        while True:
            conn_sock, conn_addr = sock.accept()
            print(f"Connected by {conn_addr}")
            msg = recv_message(conn_sock)
            action = msg.data.get('action', None)
            if action is None:
                bytes_ = recv_bytes_stream(conn_sock)
                transcription, save_path = transcribe(bytes_, AudioConfig(**msg['audio_config']), msg['topic'])
                if msg["chat_mode"]:
                    # TODO: append chat history?
                    response = {
                        'text': f"Transcription:\n{transcription}\n\nResponse:\n{get_chat_response(transcription)}",
                        'save_path': str(save_path)
                    }
                else:
                    response = {'text': transcription, 'save_path': str(save_path)}
                send_message(response, conn_sock)
                conn_sock.close()
            elif action == 'delete':
                handle_deletion(msg['save_path'])
            elif action == 'wrong':
                add_to_metadata(msg['save_path'], {'transcription_error': True})
