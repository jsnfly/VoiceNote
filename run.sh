#!/bin/bash

host_ip=$( hostname -I | cut -d ' ' -f1 ) # host_ip=127.0.0.1
port=65432
# input_device_index=9  # None

scp client.py pi:/home/pi/s2t
client_cmd="python3 /home/pi/s2t/client.py --host ${host_ip} --port ${port}"
ssh pi $client_cmd & python3 server.py && fg

# Don't forget on Pi: 

# mkdir s2t && cd s2t
# git clone https://github.com/PortAudio/portaudio.git && cd portaudio
# ./configure && make
# sudo make install
# sudo apt install portaudio19-dev
# pip3 install pyaudio

# and ssh-setup