FROM voice_note_base:latest

RUN apt update && apt -y install libasound-dev portaudio19-dev

COPY stt/requirements.txt server/stt/requirements.txt
RUN pip install --no-cache-dir -r server/stt/requirements.txt

COPY stt/stt.py server/stt/stt.py

CMD ["python", "-m", "server.stt.stt"]
