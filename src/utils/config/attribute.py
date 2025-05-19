
import inspect
from typing import Callable, Dict, Type

from utils.config.helper import serialize_object
from utils.config.schema import generate_schema_from_object


class Attribute:
    def __init__(self,
                 json_field: str = None,
                 field: str = None,
                 deserializer: Callable[[object], object] = lambda x: x,
                 serializer: Callable[[object], object] = lambda x: x,
                 required: bool = False,
                 description: str = None,
                 default: object = None,
                 type: Type = None,
                 json_type: Type = None,
                 schema: dict = None):

        self._field = field
        self._type: Type = type
        self._deserializer = deserializer

        self._json_field = json_field
        self._json_type = json_type
        self._serializer = serializer

        self._required = required
        self._default = default
        self._description = description
        self._schema = schema

        self._value = None

    # --- properties ---

    @property
    def json_key(self) -> str:
        return self._json_field or self._field

    @property
    def json_type(self) -> Type:
        if self._json_type is not None:
            return self._json_type

        if Attribute.has_schema_attributes(self._type):
            return dict

        return self._type

    @property
    def json_type_str(self) -> str:
        primitives = {
            str: 'string',
            int: 'integer',
            float: 'number',
            bool: 'boolean',
            dict: 'object',
            list: 'array',
            tuple: 'array',
            set: 'array',
            bytes: 'string',
            bytearray: 'string',
            complex: 'string',
        }
        return primitives.get(self.json_type, 'object')

    # --- public methods ---

    def deserialize(self, json_obj: dict) -> None:
        if self.json_key not in json_obj:
            raise KeyError(f"Failed to remap attribute key! Key "
                           f"'{self.json_key}' not found in dict {json_obj}")

        old_val = json_obj[self.json_key]

        mapped = self._deserializer(old_val)

        if isinstance(mapped, self._type):
            json_obj[self._field] = mapped
        elif isinstance(mapped, dict):
            json_obj[self._field] = self._type(**mapped)
        else:
            json_obj[self._field] = self._type(mapped)

        if self._json_field and self._json_field != self._field:
            del json_obj[self._json_field]

    def serialize(self, json_obj: dict):
        if self._field not in json_obj:
            raise KeyError(f"Failed to serialize attribute! "
                           f"'{self._field}' not found in dict {json_obj}")

        mapped = self._serializer(json_obj[self._field])
        if Attribute.has_schema_attributes(mapped):
            mapped = serialize_object(mapped)

        if isinstance(mapped, self.json_type):
            json_obj[self.json_key] = mapped
        else:
            json_obj[self.json_key] = self.json_type(mapped)

        if self.json_key != self._field:
            del json_obj[self._field]

    def generate_schema(self, defs: dict) -> dict:
        schema = {}

        if Attribute.has_schema_attributes(self._type):
            sub_schema_name = self._type.__name__

            if sub_schema_name not in defs:
                defs[sub_schema_name] \
                    = generate_schema_from_object(self._type, defs)

            schema['$ref'] = f"#/$defs/{sub_schema_name}"
        else:
            schema['type'] = self.json_type_str

        # todo: array, enum

        if self._default is not None:
            schema['default'] = self._default
        if self._description is not None:
            schema['description'] = self._description
        if self._schema is not None:
            schema.update(self._schema)

        return schema

    @staticmethod
    def get_schema_attributes(owner: object, create: bool = False)\
            -> Dict[str, 'Attribute']:
        if not hasattr(owner, '_schema_attributes'):
            if create:
                setattr(owner, '_schema_attributes', {})
            else:
                raise AttributeError(f"Object {owner} has no schema "
                                     "attributes!")
        return getattr(owner, '_schema_attributes')

    @staticmethod
    def has_schema_attributes(owner: object) -> bool:
        return hasattr(owner, '_schema_attributes')

    def add_to_object(self, owner: object) -> None:
        attributes = self.get_schema_attributes(owner, create=True)
        attributes[self._field] = self

    # --- build-ins ---

    def __str__(self):
        return f"Attribute({self._field}: {self._type} ({self._json_field})" \
               f"[r: {self._required}, default: {self._default}, " \
               f"desc: {self._description}])"

    def __get__(self, instance, owner):
        return self._value

    def __set_name__(self, owner: object, name: str):
        self._field = name

        if self._type is None:
            self._type = inspect.get_annotations(owner)[name]
            # check if type is subscripted generic
            if hasattr(self._type, '__origin__'):
                self._type = self._type.__origin__

        self.add_to_object(owner)
