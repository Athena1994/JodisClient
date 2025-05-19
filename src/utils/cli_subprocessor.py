import asyncio
import logging
from typing import Callable


class CLISubProcessor:
    def __init__(self):
        self._returncode = None
        self._callback = None

    @property
    def callback(self):
        """Get the callback function."""
        return self._callback

    @callback.setter
    def output_callback(self, callback: Callable[[str, bool], None]):
        """Set the callback function to be called on progress updates."""
        self._callback = callback

    async def run_command(self, command: str, *params) -> bool:
        """Run a command in a subprocess and return False if it fails."""

        command = f"{command} {' '.join(params)}"

        logging.debug(f"Running command: {command}")

        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stop = False

        async def read_stream(stream: asyncio.StreamReader,
                              error: bool = False):

            while not stop:
                line = await stream.readline()
                if not line:
                    break
                if self._callback:
                    self._callback(line.decode().strip(), error)

        await asyncio.gather(
            read_stream(process.stdout, False),
            read_stream(process.stderr, True),
        )
        await process.communicate()
        stop = True

        self._returncode = process.returncode

        return self._returncode == 0
