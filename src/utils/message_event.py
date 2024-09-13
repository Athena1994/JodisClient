
import threading


class MessageEvent:
    def __init__(self):
        self._event = threading.Event()
        self._message: object = None

    def set(self, message: object):
        self._message = message
        self._event.set()

    def get_message(self):
        return self._message

    def wait(self, timeout: float) -> object:

        if not self._event.wait(timeout):
            raise TimeoutError()

        return self._message
