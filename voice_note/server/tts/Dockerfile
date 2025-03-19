FROM voice_note_base:latest

COPY tts/requirements.txt server/tts/requirements.txt
RUN pip install --no-cache-dir -r server/tts/requirements.txt

COPY tts/Spark-TTS server/tts/Spark-TTS
COPY tts/tts.py server/tts/tts.py

CMD ["python", "-m", "server.tts.tts"]
