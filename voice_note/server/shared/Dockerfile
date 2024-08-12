# TODO: Does TTS support python 3.12?
FROM python:3.11.8-bookworm

WORKDIR /usr/src/app

COPY shared/requirements.txt server/shared/requirements.txt
RUN pip install --no-cache-dir -r server/shared/requirements.txt

COPY utils server/utils
COPY base_server.py server/base_server.py
