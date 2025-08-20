FROM voice_note_base:latest

COPY tts/requirements.txt server/tts/requirements.txt
RUN pip install --no-cache-dir -r server/tts/requirements.txt

COPY tts/tts.py server/tts/tts.py

CMD ["python", "-m", "server.tts.tts"]
