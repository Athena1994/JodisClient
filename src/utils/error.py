

from dataclasses import dataclass
import traceback


@dataclass
class Error:
    """Error class for handling errors in the application."""
    name: str
    message: str
    exception: Exception = None

    def __str__(self):
        return f"Error '{self.name}': {self.message}\n" \
               f"{self.exception}\n{traceback.format_exc()}"
