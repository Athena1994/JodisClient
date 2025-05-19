
from dataclasses import dataclass
import inspect
import logging
import threading
import traceback
from typing import Callable, Dict

import socketio


from utils.message_event import MessageEvent


@dataclass
class SocketResponse:
    event: str
    data: object


class SocketIOWrapper:

    class PatchedClient(socketio.AsyncClient):
        async def _handle_event(self, namespace: str, event: str, data: dict):
            event = data[0]
            try:
                if self._handler is not None:
                    if inspect.iscoroutinefunction(self._handler):
                        await self._handler(event, namespace, data[1])
                    else:
                        self._handler(event, namespace, data[1])

                await super()._handle_event(namespace, event, data[1:])

            except Exception as e:
                logging.error(f"Error in event handler: {e}\n"
                              f"{traceback.format_exc(e)}")

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._handler: callable = None

        @property
        def handler(self) -> callable:
            return self._handler

        @handler.setter
        def handler(self, value: callable):
            self._handler = value

    def __init__(self, timeout_s: float) -> None:
        self._socket = SocketIOWrapper.PatchedClient()
        self._socket.handler = self.event_received

        self._handlers: Dict[str, Callable[[dict], None]] = {}
        self._receive_events: set[MessageEvent] = set()

        self._listen_thread: threading.Thread = None
        self._abort = False
        self._timeout_s = timeout_s

        self._lock = threading.Lock()

    async def event_received(self, event, namespace, *args):

        if len(args) == 0:
            params = None
        elif len(args) == 1:
            params = args[0]
        else:
            params = args

        logging.info(f"Received: {event} with params: {params}")

        if event not in self._handlers:
            if len(self._receive_events) != 0:
                logging.debug(f"Setting {len(self._receive_events)}"
                              " events!")
                with self._lock:
                    for e in self._receive_events:
                        e.set(SocketResponse(event, params))
            else:
                logging.warning(f"Unhandled event: {event}")
        else:
            logging.debug(f"Calling event handler '{event}'")
            if len(args) == 0:
                res = self._handlers[event](*params)
            elif len(args) == 1:
                res = self._handlers[event](params)
            else:
                res = self._handlers[event](*params)

            if inspect.iscoroutine(res):
                await res

    async def emit(self, event: str, *args, namespace: str = "/"):
        await self._socket.emit(event, data=args, namespace=namespace)

    async def disconnect(self):
        await self._socket.disconnect()

    async def connect(self, url: str, port: int, namespace: str):
        await self._socket.connect(f'{url}:{port}',
                                   transports=['websocket'],
                                   namespaces=["/", namespace],
                                   wait_timeout=self._timeout_s)

    def is_connected(self) -> bool:
        return self._socket.connected

    def register_handler(self, event: str, callback: Callable[[dict], None]):
        if event in self._handlers:
            raise Exception(f"Overwriting handler for event {event}")
        self._handlers[event] = callback

    async def receive(self) -> SocketResponse:

        message_event = MessageEvent()

        with self._lock:
            self._receive_events.add(message_event)

        try:
            return await message_event.wait(self._timeout_s)
        finally:
            self._receive_events.remove(message_event)
