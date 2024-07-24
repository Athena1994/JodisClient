

def assert_fields_in_dict(d: dict, fields: object) -> None:
    if not isinstance(fields, list):
        fields = [fields]
    missing = [f for f in fields if f not in d]
    if len(missing) > 0:
        raise ValueError(f"Missing fields: {missing}")
