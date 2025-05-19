

from dataclasses import dataclass
import dataclasses

from jsonschema import validate

from utils.config.attribute import Attribute
from utils.config.helper import set_defaults
from utils.config.schema import generate_schema


def config(cls):

    def add_missing_attributes(cls):
        """
        Add 'Attribute' objects for non 'Attribute' class fields.
        :param cls: The class to add the attributes to.
        :return: The class with the attributes added.
        """
        attributes = Attribute.get_schema_attributes(cls, create=True)

        # add attribute for all missing fields
        for field, field_meta in cls.__dataclass_fields__.items():
            if field in attributes:
                continue

            if isinstance(field_meta.default, dataclasses._MISSING_TYPE):
                default_value = None
            else:
                default_value = field_meta.default

            attributes[field] = Attribute(
                field=field,
                type=field_meta.type,
                default=default_value,
            )

        return cls

    def wrap(cls):
        # assure that the class is a dataclass
        if not hasattr(cls, '__dataclass_fields__'):
            cls = dataclass(cls)

        # add missing attributes
        cls = add_missing_attributes(cls)

        def _wrapped_init(self, *args, **kwargs):
            # if init is called with positional args, call the original init
            if len(args) == 0:
                # validate and process kwargs against the schema
                schema = generate_schema(self)
                set_defaults(kwargs, schema)
                validate(kwargs, schema)

                # remap kwargs to the class attributes
                attributes = Attribute.get_schema_attributes(self)
                for attribute in attributes.values():
                    attribute.deserialize(kwargs)

            return self.__class__._original_init(self, *args, **kwargs)

        # replace the original __init__ method
        cls._original_init = cls.__init__
        cls.__init__ = _wrapped_init
        return cls

    if cls is None:
        return wrap
    return wrap(cls)


class schema:
    def __init__(self, schema_fct):
        self._schema_fct = schema_fct

    def __set_name__(self, owner, name):
        schemas = self.get_schemas(owner, create=True)
        schemas.append(self._schema_fct(owner))

    @staticmethod
    def get_schemas(obj, create: bool = False) -> list[dict]:
        if not hasattr(obj, '_alternative_schemas'):
            if create:
                setattr(obj, '_alternative_schemas', [])
            else:
                return []

        return getattr(obj, '_alternative_schemas')
