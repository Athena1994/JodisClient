

from pathlib import Path
from typing import Dict
import unittest

from utils.config.attribute import Attribute
from utils.config.decorator import config, schema
from utils.config.helper import serialize_object
from utils.config.schema import generate_schema


class ConfigTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def assertAttribute(self, attributes: dict, key: str, fields: dict):
        self.assertIn(key, attributes)

        obj = attributes[key]

        print(obj)

        for key, value in fields.items():
            self.assertEqual(getattr(obj, key), value, f"Key: {key}")

    def test_auto_attributes(self):
        @config
        class MockConfig:
            d: float
            a: int = 1
            b: str = "test"
            c: bool = Attribute(default=True,
                                description="A boolean attribute",
                                required=True)
            e: object = Attribute(
                default={'test': 42},
                json_field="efd",
                json_type=dict,
                required=False,
                description="A dictionary attribute"
            )

        cfg = MockConfig(3.4)

        self.assertTrue(hasattr(cfg, "_schema_attributes"))

        self.assertEqual(len(cfg._schema_attributes), 5)

        self.assertAttribute(cfg._schema_attributes, "d", {
            "_field": "d",
            "_type": float,
            "_json_field": None,
            "_required": False,
            "_default": None,
            "_description": None
            })
        self.assertAttribute(cfg._schema_attributes, "a", {
            "_field": "a",
            "_type": int,
            "_json_field": None,
            "_required": False,
            "_default": 1,
            "_description": None
            })
        self.assertAttribute(cfg._schema_attributes, "b", {
            "_field": "b",
            "_type": str,
            "_json_field": None,
            "_required": False,
            "_default": 'test',
            "_description": None
            })
        self.assertAttribute(cfg._schema_attributes, "c", {
            "_field": "c",
            "_type": bool,
            "_json_field": None,
            "_required": True,
            "_default": True,
            "_description": "A boolean attribute"
            })
        self.assertAttribute(cfg._schema_attributes, "e", {
            "_field": "e",
            "_type": object,
            "_json_type": dict,
            "_json_field": 'efd',
            "_required": False,
            "_default": {'test': 42},
            "_description": "A dictionary attribute"
            })

    def test_auto_schema(self):

        @config
        class MockConfig:
            a: int
            b: str = "test"
            c: bool = Attribute(default=True,
                                description="A boolean attribute",
                                required=True)

        cfg = MockConfig(34)

        expected_schema = {
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer"
                },
                "b": {
                    "type": "string",
                    "default": "test"
                },
                "c": {
                    "type": "boolean",
                    "default": True,
                    "description": "A boolean attribute"
                }
            },
            "required": ["c"]
        }

        self.maxDiff = None
        self.assertDictEqual(generate_schema(cfg), expected_schema)

    def test_composite_auto_schema(self):

        @config
        class MockConfigA:
            a: int = 1
            b: str = "test"
            c: bool = Attribute(default=True,
                                description="A boolean attribute",
                                required=True)

        @config
        class MockConfigB:
            e: MockConfigA
            d: int = 1
            f: MockConfigA = Attribute(default={},
                                       description="A composite attribute",
                                       required=True)

        cfg = MockConfigB(MockConfigA())

        expected_schema = {
            "type": "object",
            "properties": {
                "e": {
                    "$ref": "#/$defs/MockConfigA"
                },
                "d": {
                    "type": "integer",
                    "default": 1
                },
                "f": {
                    "$ref": "#/$defs/MockConfigA",
                    "default": {},
                    "description": "A composite attribute"
                }
            },
            "required": ["f"],
            "$defs": {
                "MockConfigA": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "integer",
                            "default": 1
                        },
                        "b": {
                            "type": "string",
                            "default": "test"
                        },
                        "c": {
                            "type": "boolean",
                            "default": True,
                            "description": "A boolean attribute"
                        }
                    },
                    "required": ["c"]
                }
            }
        }

        self.maxDiff = None
        self.assertDictEqual(generate_schema(cfg), expected_schema)

    def test_nested_attribute(self):
        """
        error reconstruction: if class B has attributes of the same nested class
        A, and A in turn explicitly defines an attribute a, initialization of B
        overwrites the value of a in A during kwargs remapping.
        """

        @config
        class MockConfigA:
            a: bool = Attribute()

        @config
        class MockConfigB:
            a: MockConfigA
            b: MockConfigA

        json = {
            "a": {
                "a": True
            },
            "b": {
                "a": False
            }
        }

        cfg = MockConfigB(**json)

        self.assertIsInstance(cfg.a, MockConfigA)
        self.assertEqual(cfg.a.a, True)
        self.assertIsInstance(cfg.b, MockConfigA)
        self.assertEqual(cfg.b.a, False)

    def test_object_instantiation(self):

        @config
        class MockConfigA:
            a: int = 1
            b: str = "test"
            c: bool = Attribute(default=True,
                                description="A boolean attribute",
                                required=True)

        @config
        class MockConfigB:
            e: MockConfigA
            d: int = 1
            f: MockConfigA = Attribute(default={},
                                       description="A composite attribute",
                                       required=True)

        json = {
            "e": {
                "a": 2,
                "b": "test2",
                "c": False
            },
            "d": 3,
            "f": {
                "a": 4,
                "b": "test4",
                "c": True
            }
        }

        cfg = MockConfigB(**json)

        self.assertIsInstance(cfg.e, MockConfigA)
        self.assertEqual(cfg.e.a, 2)
        self.assertEqual(cfg.e.b, "test2")
        self.assertEqual(cfg.e.c, False)
        self.assertEqual(cfg.d, 3)
        self.assertIsInstance(cfg.f, MockConfigA)
        self.assertEqual(cfg.f.a, 4)
        self.assertEqual(cfg.f.b, "test4")
        self.assertEqual(cfg.f.c, True)

    def test_custom_schema(self):

        @config
        class MockConfig:
            a: Dict[str, str] = Attribute(
                default={},
                schema={'additionalProperties': {'type': 'string'}})

        cfg = MockConfig()

        expected_schema = {
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "string"
                    },
                    "default": {}
                }
            },
        }

        schema = generate_schema(cfg)

        self.maxDiff = None
        self.assertDictEqual(schema, expected_schema)

    def test_json_obj_keys_error(self):

        @config
        class A:
            file: Path = Attribute('file', required=True, json_type=str)

        @config
        class B:

            name: str = Attribute(required=True)
            version: str = Attribute(required=True)

            src_dir: Path = Attribute('src-dir', default='./src', json_type=str)

            requirements_file: Path = Attribute(
                'requirements', default='./requirements.txt', json_type=str)

            entry_point: A = Attribute('entry-point', required=True)

        json = {
            "name": "test",
            "version": "1.0",
            "src-dir": "./src",
            "requirements": "./requirements.txt",
            "entry-point": {
                "file": "./test.py",
            }
        }

        cfg = B(**json)

        json_obj = serialize_object(cfg)

        expected_json = {
            "name": "test",
            "version": "1.0",
            "src-dir": "src",
            "requirements": "requirements.txt",
            "entry-point": {
                "file": "test.py"
            }
        }
        self.assertDictEqual(json_obj, expected_json)

    def test_nested_json_error(self):
        @config
        class A:
            a: str = Attribute(
                'a-json', json_type=dict, deserializer=lambda x: x['dist'],
                required=True, serializer=lambda x: {"dist": int(x)})
            b: list = Attribute('b-json')
            c: dict = Attribute('c-json')

        json_obj = {
            "a-json": {
                "dist": 42,
            },
            "b-json": [1, 2, 3],
            "c-json": {
                "key": "value"
            }
        }

        cfg = A(**json_obj)
        self.assertEqual(cfg.a, "42")
        self.assertEqual(cfg.b, [1, 2, 3])
        self.assertEqual(cfg.c, {"key": "value"})

        json_obj = serialize_object(cfg)

        expected_json = {
            "a-json": {
                "dist": 42,
            },
            "b-json": [1, 2, 3],
            "c-json": {
                "key": "value"
            }
        }
        self.assertDictEqual(json_obj, expected_json)

    def test_shorthand_error(self):
        @config
        class B:
            a: str = Attribute(required=True)
            b: str = Attribute(required=True)

            @schema
            def alternative_schema(self):
                return {
                    "type": "string"
                }

        @config
        class A:
            a: B = Attribute(
                json_field='a-json',
                deserializer=lambda x: (
                    {'a': x, 'b': x} if isinstance(x, str) else x),
                required=True
            )

        json_obj = {
            "a-json": {
                "a": "testA",
                "b": "testB"
            }
        }
        cfg = A(**json_obj)
        self.assertEqual(cfg.a.a, "testA")
        self.assertEqual(cfg.a.b, "testB")

        expected_schema = {
            "type": "object",
            "properties": {
                "a-json": {"$ref": "#/$defs/B"},
            },
            "required": ["a-json"],
            "$defs": {
                "B": {
                    "oneOf": [
                        {
                            "type": "string"
                        }, {
                            "type": "object",
                            "properties": {
                                "a": {
                                    "type": "string"
                                },
                                "b": {
                                    "type": "string"
                                }
                            },
                            "required": ["a", "b"]
                        }
                    ]
                }
            }
        }
        self.maxDiff = None
        gschema = generate_schema(cfg)
        self.assertDictEqual(gschema, expected_schema)

        json_obj_new = serialize_object(cfg)
        self.assertDictEqual(json_obj_new, json_obj)

        json_obj = {
            "a-json": "testA"
        }
        cfg = A(**json_obj)
        self.assertEqual(cfg.a.a, "testA")
        self.assertEqual(cfg.a.b, "testA")
        json_obj_new = serialize_object(cfg)
        self.assertDictEqual(json_obj_new, {
            "a-json": {
                "a": "testA",
                "b": "testA"
            }
        })
