"""
    This is the client worker script. It is provided with a json-file that
    contains instructions about a training session. The worker will prepare the
    necessary data and training infrastructure and start the training session.
"""

import argparse
from dataclasses import dataclass
import logging
import time
from program.base_program import BaseProgram
from program.job.job_packer import JobExtractor
from program.training_manager import TrainingManager, TrainingReporter
from utils.config_utils import assert_fields_in_dict
from utils.path import TmpDir

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

    parser = argparse.ArgumentParser(description="Worker script for training.")
    parser.add_argument("job", help="Path to job archive.")
    parser.add_argument("--update", "-u", default=1000,
                        help="ms between updates.")
    parser.add_argument("--log", "-l", default="worker.log",
                        help="ms between updates.")

    args = parser.parse_args()

    update_time = float(args.update) / 1000

    logging.basicConfig(filename=args.log,
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info("Starting worker script...")
    try:
        with TmpDir() as tmp_dir:
            ex = JobExtractor(args.job, tmp_dir)

            program = BaseProgram(ex.get_config())

        program.start_training_async()

        training_info = program._trainer.get_info()

        while program.is_running():
            time.sleep(update_time)

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
