"""
    This is the client worker script. It is provided with a json-file that
    contains instructions about a training session. The worker will prepare the
    necessary data and training infrastructure and start the training session.
"""

import argparse
from dataclasses import dataclass
import json
import logging
import os
import shutil
import tempfile
import time
import zipfile

from program.base_program import BaseProgram
from program.config_loader import ConfigFiles
from program.training_manager import TrainingManager, TrainingReporter
from utils.config_utils import assert_fields_in_dict

logging.basicConfig(filename="worker.log",
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class JobConfig:
    cfg: ConfigFiles

    @staticmethod
    def from_dict(d: dict):
        assert_fields_in_dict(d, ['files'])
        return JobConfig(ConfigFiles.from_dict(d['files']))


class TmpDir:
    def __init__(self):
        self.path = None

    def __enter__(self):
        self.path = tempfile.mkdtemp()
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.path)


def load_job(tmp_dir: str, file: str) -> JobConfig:
    if not os.path.isfile(file):
        raise FileNotFoundError("Job archive is not found or not a file.")

    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(tmp_dir)

    job_cfg_file = os.path.join(tmp_dir, 'job.json')

    if not os.path.isfile(job_cfg_file):
        raise FileNotFoundError("job.json not found in archive.")

    with open(job_cfg_file, 'r') as f:
        job = JobConfig.from_dict(json.load(f))

    job.cfg = job.cfg.with_base_path(tmp_dir)

    return job


@dataclass
class JobReport:
    @dataclass
    class Progress:
        ix: int
        cnt: int
        av_time: float
        last_time: float

    episode: Progress
    tr_progress: Progress
    val_progress: Progress

    last_evaluation: dict

    def to_dict(self):
        res = self.__dict__.copy()
        res['tr_progress'] = self.tr_progress.__dict__
        res['val_progress'] = self.val_progress.__dict__
        return res

    @staticmethod
    def from_dict(d: dict):
        assert_fields_in_dict(d, ['episode', 'tr_progress', 'val_progress',
                                  'last_evaluation'])
        return JobReport(
            JobReport.Progress(**d['episode']),
            JobReport.Progress(**d['tr_progress']),
            JobReport.Progress(**d['val_progress']),
            d['last_evaluation']
        )

    @staticmethod
    def from_state(state: TrainingReporter.State, info: TrainingManager.Info):
        return JobReport(
            JobReport.Progress(state.epoch.ix,
                               info.max_epoch,
                               state.epoch.av_time,
                               state.epoch.last_time),
            JobReport.Progress(
                state.training.ix / info.experiences_per_step,
                info.training_samples_cnt / info.experiences_per_step,
                state.training.av_time, None),
            JobReport.Progress(
                state.validation.ix,
                info.validation_sample_cnt,
                state.validation.av_time, None),
            state.last_evaluation
        )


def main():

    update_time_ms = 1000
    job_file = "examples/example_job.zip"

    logging.info("Starting worker script...")

    parser = argparse.ArgumentParser(description="Worker script for training.")
    parser.add_argument("job", help="Path to job archive.")
    parser.add_argument("--update", "-u", default=1000,
                        help="ms between updates.")

    args = parser.parse_args()
    job_file = args.job
    update_time_ms = int(args.update)

    try:
        with TmpDir() as tmp_dir:
            job = load_job(tmp_dir, job_file)
            program = BaseProgram(job.cfg)

        program.start_training_async()

        training_info = program._trainer.get_info()

        while program.is_running():
            time.sleep(update_time_ms / 1000)

            tr_state = program._trainer._reporter.get_state()
            report = JobReport.from_state(tr_state, training_info)
            print(report.to_dict())

    except SystemExit as e:
        logging.error(str(e))
        return
    except FileNotFoundError as e:
        logging.error("Error loading job: " + str(e))
        return
    except Exception as e:
        logging.exception(f"Error loading job: {e}")
        return


if __name__ == '__main__':
    main()
