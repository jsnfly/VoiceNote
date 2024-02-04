FROM voice_note_base:latest

COPY chat/requirements.txt server/chat/requirements.txt
RUN pip install --no-cache-dir -r server/chat/requirements.txt

COPY chat/chat.py server/chat/chat.py

CMD ["python", "-m", "server.chat.chat"]
