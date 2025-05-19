

from abc import abstractmethod
from asyncio import Event
from core.http_api.api_objects import JobDO
from utils.cache import Cache


class ClientModuleControl:
    def __init__(self):
        self._busy = False
        self._abort_event = Event()
        self._finished_event = Event()

    # --- Abstract Methods ---

    @abstractmethod
    async def prepare(self, job: dict, cache: Cache) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def run(self, job: dict, cache: Cache) -> None:
        raise NotImplementedError()

    # --- Properties ---

    @property
    def abort_event(self) -> Event:
        return self._abort_event

    @property
    def finished_event(self) -> Event:
        return self._finished_event

    @property
    def abort_requested(self) -> bool:
        return self._abort_event.is_set()

    @property
    def is_busy(self) -> bool:
        return self._busy

    # --- Public Methods ---

    async def start(self, job: JobDO, cache: Cache) -> None:
        self._abort_event.clear()
        self._finished_event.clear()
        self._busy = True
        try:
            await self.prepare(job, cache)
            await self.run(job, cache)
        except Exception as e:
            raise e
        finally:
            self._busy = False
            self._finished_event.set()

    async def abort(self, block: bool = True) -> None:
        self._abort_event.set()
        if block:
            await self._finished_event.wait()
