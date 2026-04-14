FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

WORKDIR /usr/src/app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip build-essential && rm -rf /var/lib/apt/lists/*

COPY chat/requirements.txt server/chat/requirements.txt
RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 python3 -m pip install --break-system-packages --no-cache-dir -r server/chat/requirements.txt

COPY utils server/utils
COPY base_server.py server/base_server.py
COPY __init__.py server/__init__.py

COPY chat/chat.py server/chat/chat.py
COPY chat/tools.py server/chat/tools.py

CMD ["python3", "-m", "server.chat.chat"]
