"""
    This is the client worker script. It is provided with a json-file that
    contains instructions about a training session. The worker will prepare the
    necessary data and training infrastructure and start the training session.
"""

import argparse
import logging
import time
from program.base_program import BaseProgram
from program.job.job_packer import JobExtractor
from program.job.job_report import JobReport
from utils.path import TmpDir


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
            print(report.to_dict(), flush=True)

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
