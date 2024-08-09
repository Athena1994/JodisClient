
from dataclasses import dataclass
from program.config_loader import ConfigFiles
from utils.config_utils import assert_fields_in_dict


@dataclass
class JobConfig:
    version: str
    cfg: ConfigFiles

    @staticmethod
    def from_dict(d: dict):

        assert_fields_in_dict(d, ['version', 'files'])
        return JobConfig(
            d['version'],
            ConfigFiles.from_dict(d['files']))
