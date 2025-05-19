import logging
from pathlib import Path
import re
from types import CoroutineType
from typing import Callable, List, Tuple

from utils.cli_subprocessor import CLISubProcessor


class PipRequirementsControl:

    def __init__(self):
        self._cli = CLISubProcessor()

    @property
    def cli(self):
        """Get the CLI subprocess processor."""
        return self._cli

    @staticmethod
    def _analyse_requirements(file: Path) -> List[str]:
        if not file.exists():
            raise ValueError(f"Requirements file {file} does not "
                             "exist.")

        requirements = list(map(
            lambda r: next(re.finditer(r'^[^=><]*', r.strip())).group(0),
            filter(lambda f: f.strip() != "",
                   file.read_text().splitlines())))

        logging.debug(f"found {len(requirements)} requirements in {file}")

        return requirements

    def install_requirements(self,
                             requirements_file: Path,
                             dev: bool = False) -> 'InstallProgress':
        """Check if requirements file is met."""

        self._analyse_requirements(requirements_file)

        prog = InstallProgress()

        self.cli.output_callback = prog.feed_line

        prog.coroutine = self.cli.run_command(
            "pip",
            "install",
            "--dry-run" if dev else "",
            f"-r {requirements_file}"
        )
        return prog


class InstallProgress:

    class UpdateType:
        ERROR = 0
        BEGIN = 1
        PROGRESS = 2
        FINISHED = 3

    def __init__(self):
        self._current = ''

        self._error = None

        self._callback = None

        self._coroutine: CoroutineType = None

    def feed_line(self, line: str, error: bool):

        def parse(line: str) -> Tuple[str, InstallProgress.UpdateType]:
            """Parse the line and update the progress."""

            match = re.search(r'Downloading (\S+)\b', line)
            if match:
                return (match.group(1), InstallProgress.UpdateType.FINISHED)

            match = re.search(r'Collecting (\S+)\b', line)
            if match:
                return (match.group(1), InstallProgress.UpdateType.BEGIN)

            match = re.search(r'Using cached (.*?) ', line)
            if match:
                return (match.group(1), InstallProgress.UpdateType.FINISHED)

            match = re.search(r'Requirement already satisfied: (.*)? ', line)
            if match:
                return (match.group(1), InstallProgress.UpdateType.FINISHED)

            print("DEBUG: ", line)
            return None

        if error:
            if self._error is None:
                self._error = ""
            self._error = '\n'.join([self._error, line])
            update_type = InstallProgress.UpdateType.ERROR
        else:
            msg = parse(line)
            if msg is None:
                return
            self._current = msg[0]
            update_type = msg[1]

        if self._callback:
            self._callback(self, update_type)

    @property
    def error(self):
        return self._error

    @property
    def callback(self):
        return self._callback

    @callback.setter
    def callback(self,
                 callback: Callable[['InstallProgress', UpdateType], None]):
        """Set the callback function to be called on progress updates."""
        self._callback = callback

    @property
    def current_requirement(self):
        return self._current

    @property
    def coroutine(self):
        return self._coroutine

    @coroutine.setter
    def coroutine(self, coroutine: CoroutineType):
        self._coroutine = coroutine

    @property
    def all_installed(self):
        return not self._coroutine.cr_running and not self.failed

    @property
    def finished(self):
        return not self._coroutine.cr_running

    @property
    def failed(self):
        return self._error is not None
