from _hashlib import HASH as Hash
import hashlib
import os


def md5_hash_dir(path: str) -> Hash:
    h = hashlib.md5()
    for root, _, files in os.walk(path):
        if root.endswith("__pycache__"):
            continue
        for file in files:
            with open(os.path.join(root, file), 'rb') as f:
                h.update(f.read())
    return h


def hash_file(path: str, prev_hash: Hash = None) -> Hash:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File '{path}' not found!")

    if prev_hash is not None:
        h = prev_hash
    else:
        h = hashlib.md5()

    with open(path, 'rb') as f:
        h.update(f.read())
    return h
