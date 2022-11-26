import ast
import json

# TODO: Type hints would make this easier to understand.
class Message:
    SEP = "\n\n\n".encode()

    def __init__(self, data):
        self.data = data

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
            except ValueError:
                return val

        return cls._apply_recursively_to_type(_func, str, data.copy())

    @classmethod
    def _apply_recursively_to_type(cls, func, target_type, dict_node):
        for key, val in dict_node.items():
            if isinstance(val, dict):
                cls._apply_recursively_to_type(func, target_type, val)
            elif isinstance(val, target_type):
                dict_node[key] == func(val)
        return dict_node


def send_message(data, socket):
    socket.sendall(Message(data).encode())

# TODO: refactor
def recv_message(socket):
    assert not socket.getblocking(), "Only non-blocking sockets are supported."

    msg_bytes, other_bytes = [], []
    msg_has_started = False
    while True:
        try:
            bytes_ = socket.recv(2**20)
        except BlockingIOError:
            continue
        sep_count = bytes_.count(Message.SEP)
        if sep_count == 1:
            if msg_has_started:
                msg_b, other_b = bytes_.split(Message.SEP)
                msg_bytes.append(msg_b)
                other_bytes.append(other_b)
                msg = Message.decode(b''.join(msg_bytes))
                return msg, b''.join(other_bytes)
            else:
                msg_has_started = True
                other_b, msg_b = bytes_.split(Message.SEP)
                msg_bytes.append(msg_b)
                other_bytes.append(other_b)
        elif sep_count == 2:
            other1, msg, other2 = bytes_.split(Message.SEP)
            assert len(other1) == len(other2) == 0
            return Message.decode(msg), b''
        elif sep_count == 0:
            if msg_has_started:
                msg_bytes.append(bytes_)
            else:
                return None, bytes_
        else:
            raise Exception('More than one Message in buffer is currently not supported')
