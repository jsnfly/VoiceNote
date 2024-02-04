#!/usr/bin/env bash

# https://stackoverflow.com/a/246128
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

docker build -t voice_note_base:latest -f ${SCRIPT_DIR}/shared/Dockerfile ${SCRIPT_DIR}

docker compose -f ${SCRIPT_DIR}/compose.yml up -d --build
