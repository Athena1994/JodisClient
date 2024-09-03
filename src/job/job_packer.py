
from io import TextIOWrapper
import json
import os
import zipfile

from core.config_loader import ConfigFiles


class JobPacker:

    def __init__(self):
        pass


class JobExtractor:

    class FileNames:
        JOB = 'job.json'

    def _extract_file(self, file: str, dir: str | None = None):
        with zipfile.ZipFile(self._zip_file, "r") as zip_ref:
            if file not in [f.filename for f in zip_ref.filelist]:
                raise FileNotFoundError(f"{file} not found in archive.")

            if dir is not None:
                zip_ref.extract(file, dir)
                return os.path.join(dir, file)
            else:
                return zip_ref.open(file)

    def __init__(self, file: str, working_dir: str):

        self._zip_file = file

        with TextIOWrapper(self._extract_file(JobExtractor.FileNames.JOB)) as f:
            self.job = JobConfig.from_dict(json.load(f))

        self._extract_file(self.job.cfg.agent, working_dir)
        self._extract_file(self.job.cfg.data, working_dir)
        self._extract_file(self.job.cfg.training, working_dir)
        self._extract_file(self.job.cfg.simulation, working_dir)
        self._extract_file(self.job.cfg.evaluation, working_dir)

        self.job.cfg = self.job.cfg.with_base_path(working_dir)

    def get_config(self) -> ConfigFiles:
        return self.job.cfg
