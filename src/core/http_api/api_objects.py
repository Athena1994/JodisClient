

from dataclasses import dataclass
import datetime
import json


@dataclass
class ModuleDO:
    name: str
    id: str
    version: str
    hash: str
    client_url: str
    config_component: object

    @staticmethod
    def from_json(json_str: str):
        """
        Convert a JSON string to a ModuleDO object.
        :param json_str: The JSON string to convert.
        :return: The ModuleDO object.
        """
        return ModuleDO(**json.loads(json_str))


@dataclass
class JobDO:
    id: int
    client_id: str

    module_id: str
    module_name: str

    name: str
    config: dict

    rank: int
    state: str
    sub_state: str

    timestamp: datetime.datetime

    @staticmethod
    def from_json(json_str: str):
        return JobDO(**json.loads(json_str))
