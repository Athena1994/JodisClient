import hashlib


def get_module_id(module_name: str, module_version: str) -> str:
    return hashlib.sha256(
        f"{module_name}:{module_version}".encode()).hexdigest()
