"""Serialise complex Python types to JSON using pickled hex.

- Serialises complex values (list, dicts, classes) as binary strings, pickled and encoded as hex.

- These values are not readable at the frontend. If you want to have frontend readable values,
  store human-readable copies
"""
import pickle


def encode_pickle_over_json(v: any):
    pickled = pickle.dumps(v)
    return pickled.hex()


def decode_pickle_over_json(v: str) -> any:
    bin = bytearray.fromhex(v)
    unpickled = pickle.loads(bin)
    return unpickled
