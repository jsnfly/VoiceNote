import ast
import time
import json

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
        return self._apply_recursively_to_type(str, bytes, data.copy())

    @classmethod
    def _destringify(cls, data):
        def _func(val):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return val

        return cls._apply_recursively_to_type(_func, str, data.copy())

    @classmethod
    def _apply_recursively_to_type(cls, func, target_type, dict_node):
        for key, val in dict_node.items():
            if isinstance(val, dict):
                cls._apply_recursively_to_type(func, target_type, val)
            elif isinstance(val, target_type):
                dict_node[key] = func(val)
        return dict_node


# TODO: own Socket class?
def send_message(data, socket):
    send_data(Message(data).encode(), socket)

# TODO: always send messages (get rid of `send_data``)?
def send_data(data, socket):
    bytes_sent = 0
    while bytes_sent < len(data):
        try:
            bytes_sent += socket.send(data[bytes_sent:])
        except BlockingIOError:
            # Send buffer is full, wait a bit and try again.
            time.sleep(0.02)


def _ends_with_incomplete_separator(bytes_):
    if bytes_.endswith(Message.SEP):
        return False
    return bytes_.endswith(tuple(bytes([i]) for i in Message.SEP))


def recv_messages(socket, blocking=True):

    # Socket is supposed to be non-blocking, because otherwise `socket.recv` will block until it receives at least one
    # byte.
    # Good resource: https://docs.python.org/3/howto/sockets.html#socket-howto
    assert not socket.getblocking(), "Only non-blocking sockets are supported." # TODO: is this necessary?

    bytes_ = b''
    while True:
        try:
            received = socket.recv(4096)
            bytes_ += received  # May be slow because bytes are immutable, but should not matter.
            if len(received) == 4096:  # TODO: if len(received) == 0:  # Connection closed?
                continue
        except BlockingIOError:
            # Nothing to receive.
            if blocking:
                time.sleep(0.02)
                continue
            else:
                return [], bytes_
        splits = bytes_.split(Message.SEP)
        if not (len(splits) % 2 == 0 or _ends_with_incomplete_separator(bytes_)):
            break
    messages = [splits[i] for i in range(1, len(splits) - 1) if splits[i-1] == splits[i+1] == b'']
    other = [sp for sp in splits if len(sp) > 0 and sp not in messages]

    return [Message.decode(msg) for msg in messages], b''.join(other)
