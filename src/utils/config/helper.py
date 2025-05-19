
import copy
import json
from jsonschema import validate
from pathlib import Path
from typing import Dict, Type, TypeVar

from utils.config.schema import generate_schema


T = TypeVar('T')


def set_defaults(json_obj: dict, schema: dict, defs: dict = None):
    if defs is None:
        defs = schema.get('$defs', {})

    # if sub-schema is referenced, get the schema from the definitions
    ref: str = schema.get('$ref')
    if ref is not None:
        ref = ref.split('/')[-1]
        schema = defs.get(ref)

    # default will be set by property owner
    if schema.get('type') != 'object':
        return

    properties_schema: Dict[str, dict] = schema.get('properties', {})

    # find missing keys
    missing_keys = set(properties_schema.keys()) - set(json_obj.keys())
    for missing_key in missing_keys:
        # check if default exists
        sub_schema = properties_schema[missing_key]
        if 'default' not in sub_schema:
            continue
        # generate default value from schema and add to json object
        default_value = sub_schema['default']
        set_defaults(default_value, sub_schema, defs)
        json_obj[missing_key] = default_value


def serialize_dict(obj: dict) -> dict:
    """
    Serialize a dictionary to a JSON-compatible format.
    :param obj: The dictionary to serialize.
    :return: A JSON-compatible dictionary.
    """
    if isinstance(obj, dict):
        return {k: serialize_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_dict(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return serialize_object(obj)
    else:
        return obj


def serialize_object(obj: object) -> dict:
    from utils.config.attribute import Attribute

    json_obj = copy.deepcopy(obj.__dict__)

    for attribute in Attribute.get_schema_attributes(obj).values():
        attribute.serialize(json_obj)

    return json_obj


def load_config(path: Path, type: Type[T]) -> T:
    """
    Load a configuration from a JSON file.
    :param path: The path to the JSON file.
    :param type: The type of the configuration class.
    :return: An instance of the configuration class.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file {path} does not exist.")

    try:
        with open(path, 'r') as f:
            return type(**json.load(f))
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON file {path}: {e}")


def save_config(config: object, path: Path):
    """
    Save a configuration to a JSON file.
    :param config: The configuration object.
    :param path: The path to the JSON file.
    """
    json_obj = serialize_object(config)

    validate(json_obj, generate_schema(config))

    with open(path, 'w') as f:
        json.dump(json_obj, f, indent=4)
