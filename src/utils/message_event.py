
import asyncio


class MessageEvent:
    def __init__(self):
        self._event = asyncio.Event()
        self._message: object = None

    def set(self, message: object):
        self._message = message
        self._event.set()

    def get_message(self):
        return self._message

    async def wait(self, timeout: float) -> object:

        waiter = self._event.wait()

        asyncio.get_event_loop().call_later(
            delay=timeout,
            callback=lambda: self.set(None))

        await waiter

        if self._message is None:
            raise TimeoutError()

        return self._message
