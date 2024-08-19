
from dataclasses import dataclass
from uuid import UUID
from program.config_loader import ConfigFiles
from utils.config_utils import assert_fields_in_dict


@dataclass
class JobConfig:
    id: UUID
    version: str
    cfg: ConfigFiles

    @staticmethod
    def from_dict(d: dict):

        assert_fields_in_dict(d, ['version', 'files', 'id'])
        return JobConfig(
            UUID(d['id']),
            d['version'],
            ConfigFiles.from_dict(d['files']))
