
import json
import os
import subprocess
import threading

from program.job.job_report import JobReport
from utils.path import TmpDir


class JobManager:
    def __init__(self, update_rate: float):
        self._update_rate = update_rate
        self._working_dir = TmpDir()
        self._working_dir.create()
        self._worker_proc = None
        self._worker_log_file = os.path.join(self._working_dir.get(),
                                             "worker.log")
        self._worker_out = None
        self._worker_state = JobReport(
            JobReport.Progress(0, 0, 0, 0),
            JobReport.Progress(0, 0, 0, 0),
            JobReport.Progress(0, 0, 0, 0),
            {}
        )

    def spawn_worker(self, job_file: str):
        self._worker_proc = subprocess.Popen(
            ["python3", "program/job/job_worker.py",
             job_file,
             "-l", self._worker_log_file,
             "-u", str(self._update_rate)],
            stdout=subprocess.PIPE)

        self._worker_out = self._worker_proc.stdout

        def read_worker_output():
            while True:
                line = self._worker_out.readline()
                if not line:
                    break
                line = str(line)[2:-3].replace('\'', '\"')
                self._worker_state \
                    = JobReport.from_dict(json.loads(line))

        threading.Thread(target=read_worker_output).start()

    def get_job_state(self):
        return self._worker_state

    def stop_worker(self):
        if self._worker_proc is not None:
            self._worker_proc.terminate()
            self._worker_proc.wait()
        self._working_dir.close()
