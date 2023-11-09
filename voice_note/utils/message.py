import json
import base64


# TODO: Type hints would make this easier to understand.
class Message:
    SEP = "\n\n\n\n\n\n\n\n\n".encode()  # Can't be too short or else it will appear by chance.

    def __init__(self, data):
        self.data = data

    def __contains__(self, key):
        return self.data.__contains__(key)

    def __getitem__(self, key):
        return self.data[key]

    def encode(self):
        encoded = json.dumps(self._stringify(self.data)).encode()
        return self.SEP + encoded + self.SEP

    @classmethod
    def decode(cls, bytes_):
        bytes_ = bytes_.strip(cls.SEP)
        return cls(cls._destringify(json.loads(bytes_.decode())))

    def _stringify(self, data):
        transformed = {}
        for key, val in data.items():
            if isinstance(val, dict):
                transformed[key] = self._stringify(val)
            elif isinstance(val, bytes):
                transformed[key + '_base64'] = base64.b64encode(val).decode()
            else:
                transformed[key] = val
        return transformed

    @classmethod
    def _destringify(cls, data):
        transformed = {}
        for key, val in data.items():
            if isinstance(val, dict):
                transformed[key] = cls._destringify(val)
            elif key.endswith('_base64'):
                transformed[key.removesuffix('_base64')] = base64.b64decode(val)
            else:
                transformed[key] = val
        return transformed


def send_message(data, sock):
    sock.sendall(Message(data).encode())


def _ends_with_incomplete_separator(bytes_):
    if bytes_.endswith(Message.SEP):
        return False
    return bytes_.endswith(tuple(bytes([i]) for i in Message.SEP))


def recv_message(sock):
    message = b''
    while len(message.split(Message.SEP)) != 3:
        message += sock.recv(4096)
    return Message.decode(message)


def recv_bytes_stream(sock):
    bytes_ = []
    while (received := sock.recv(4096)) != b'':
        bytes_.append(received)
    return b''.join(bytes_)
