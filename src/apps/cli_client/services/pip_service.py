

import asyncio
from pathlib import Path

from errors.invalid_state_error import InvalidStateError
from utils.requirements_check import InstallProgress, PipRequirementsControl


class PipInstallError(Exception):
    """
    Exception raised when there is an error in the pip service.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"PipServiceError: {self.message}"


class PipService:
    def __init__(self):
        self._pip = PipRequirementsControl()

        self._current_progress = None

        self._lock = asyncio.Lock()

    @property
    def busy(self) -> bool:
        """
        Check if the pip service is busy.
        :return: True if the pip service is busy, False otherwise.
        """
        return self._current_progress is not None and \
            not self._current_progress.finished

    @property
    def installation_info(self) -> InstallProgress:
        return self._current_progress

    async def install(self, requirements_file: Path, block: bool,
                      raise_on_error: bool) \
            -> InstallProgress:

        if not block and self._lock.locked():
            raise InvalidStateError("Pip service is busy.")

        async with self._lock:
            prog = self._pip.install_requirements(requirements_file)
            self._current_progress = prog

            await self._current_progress.coroutine

            if prog.error and raise_on_error:
                raise PipInstallError(f"Installation failed: {prog.error}")

        return prog
