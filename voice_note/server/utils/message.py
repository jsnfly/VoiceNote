import json
import base64
from typing import Dict, Union, Any

# Define recursive types.
DataValue = Union[str, bytes, int, float, bool, None, "DataDict"]
DataDict = Dict[str, DataValue]
EncodedValue = Union[str, int, float, bool, None, "EncodedDict"]
EncodedDict = Dict[str, EncodedValue]


def _stringify_values(data: DataDict) -> EncodedDict:
    transformed = {}
    for key, val in data.items():
        if isinstance(val, dict):
            transformed[key] = _stringify_values(val)
        elif isinstance(val, bytes):
            transformed[key + '_base64'] = base64.b64encode(val).decode()
        else:
            transformed[key] = val
    return transformed


def encode(data: DataDict) -> str:
    """Encodes a dictionary to a JSON string, with bytes converted to base64."""
    return json.dumps(_stringify_values(data))


def _destringify_values(encoded_data: EncodedDict) -> DataDict:
    transformed = {}
    for key, val in encoded_data.items():
        if isinstance(val, dict):
            transformed[key] = _destringify_values(val)
        elif isinstance(key, str) and key.endswith('_base64'):
            transformed[key.removesuffix('_base64')] = base64.b64decode(val)
        else:
            transformed[key] = val
    return transformed


def decode(data: str) -> DataDict:
    """Decodes a JSON string into a dictionary, with base64 values converted to bytes."""
    return _destringify_values(json.loads(data))
