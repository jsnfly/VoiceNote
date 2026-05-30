FROM node:22-bookworm

WORKDIR /usr/src/app

ENV DEBIAN_FRONTEND=noninteractive
ENV PI_SKIP_VERSION_CHECK=1
ENV PI_TELEMETRY=0

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*

COPY package.json package.json
RUN npm install --omit=dev

COPY chat/requirements.txt server/chat/requirements.txt
RUN python3 -m pip install --break-system-packages --no-cache-dir -r server/chat/requirements.txt
RUN mkdir -p /usr/src/app/pi-agent /usr/src/app/logs && chown -R 1000:1000 /usr/src/app/pi-agent /usr/src/app/logs

COPY utils server/utils
COPY base_server.py server/base_server.py
COPY __init__.py server/__init__.py

COPY chat/chat.py server/chat/chat.py

CMD ["python3", "-m", "server.chat.chat"]
