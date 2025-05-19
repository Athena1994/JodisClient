

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Type


def load_module(file: Path, module_key: str):
    if not file.exists():
        raise ValueError(f"Control path {file} does not exist.")

    spec = importlib.util.spec_from_file_location(
        module_key, file)
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def instanciate_class(module: ModuleType, class_name: str,
                      args: list = [], kwargs: dict = {}):
    if not hasattr(module, class_name):
        raise ValueError(f"Class {class_name} not found in module"
                         f"{module.__name__}.")

    class_type: Type = getattr(module, class_name)
    return class_type(*args, **kwargs)
