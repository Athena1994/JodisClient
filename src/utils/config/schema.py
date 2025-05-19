
def update_key(target: dict, key: str, value: object) -> None:
    """
    Update a key in a dictionary, ensuring nested dictionaries are not
    overwritten.
    """
    current_value = target.get(key)

    if current_value is None or not isinstance(current_value, dict) \
            or not isinstance(value, dict):
        target[key] = value
    else:
        for inner_key, inner_val in value.items():
            update_key(target[key], inner_key, inner_val)


def generate_schema(obj: object) -> dict:
    """
    Generate a JSON schema from 'Attribute' members of object.
    :return: A dictionary representing the JSON schema.
    """

    defs = {}

    schema = generate_schema_from_object(obj, defs)

    if len(defs) > 0:
        schema['$defs'] = defs

    return schema


def generate_schema_from_object(obj: object, defs: dict) -> dict:

    """
    Generate a JSON schema from 'Attribute' members of object.
    :return: A dictionary representing the JSON schema.
    """

    from utils.config.attribute import Attribute

    schema_attributes = Attribute.get_schema_attributes(obj)

    alternative_schemas: list[dict] = getattr(obj, '_alternative_schemas', None)

    schema = {}

    schema['type'] = 'object'
    update_key(schema, 'properties', {})

    required = []

    for attribute in schema_attributes.values():
        update_key(schema['properties'], attribute.json_key,
                   attribute.generate_schema(defs))
        if attribute._required:
            required.append(attribute.json_key)

    if len(required) > 0:
        schema['required'] = required

    if alternative_schemas is not None:
        return {
            'oneOf': alternative_schemas+[schema]
        }

    return schema
