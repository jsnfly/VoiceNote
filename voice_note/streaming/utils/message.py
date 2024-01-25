import json
import base64
from typing import Dict, Union


# TODO: should this be a class?
class Message:

    # Define recursive types.
    DataValue = Union[str, bytes, "DataDict"]
    DataDict = Dict[str, DataValue]
    EncodedValue = Union[str, "EncodedDict"]
    EncodedDict = Dict[str, EncodedValue]

    def __init__(self, data: DataDict):
        self.data = data

    def __contains__(self, key: str) -> bool:
        return self.data.__contains__(key)

    def __getitem__(self, key: str) -> DataValue:
        return self.data[key]

    def encode(self) -> str:
        return json.dumps(self._stringify_values(self.data))

    def _stringify_values(self, data: DataDict) -> EncodedDict:
        transformed = {}
        for key, val in data.items():
            if isinstance(val, dict):
                transformed[key] = self._stringify_values(val)
            elif isinstance(val, bytes):
                transformed[key + '_base64'] = base64.b64encode(val).decode()
            else:
                transformed[key] = val
        return transformed

    @classmethod
    def from_data_string(cls, data: str) -> 'Message':
        return cls(cls.decode(data))

    @classmethod
    def decode(cls, data: str) -> DataDict:
        return cls._destringify_values(json.loads(data))

    @classmethod
    def _destringify_values(cls, encoded_data: EncodedDict) -> DataDict:
        transformed = {}
        for key, val in encoded_data.items():
            if isinstance(val, dict):
                transformed[key] = cls._destringify_values(val)
            elif key.endswith('_base64'):
                transformed[key.removesuffix('_base64')] = base64.b64decode(val)
            else:
                transformed[key] = val
        return transformed
