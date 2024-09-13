
from dataclasses import dataclass
import logging
import threading
from typing import Callable, Dict
import socketio

from src.utils.message_event import MessageEvent


@dataclass
class SocketResponse:
    event: str
    data: object


class SocketIOWrapper:

    def __init__(self, timeout_s: float) -> None:
        self._socket = socketio.Client()
        self._socket.on('*', self.event_received, '*')

        self._handlers: Dict[str, Callable[[dict], None]] = {}
        self._receive_events: set[MessageEvent] = set()

        self._listen_thread: threading.Thread = None
        self._abort = False
        self._timeout_s = timeout_s

        self._lock = threading.Lock()

    def event_received(self, event, namespace, *args):
        if len(args) == 0:
            params = None
        elif len(args) == 1:
            params = args[0]
        else:
            params = args

        logging.debug(f"Received: {event} with params: {params}")

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
                self._handlers[event]()
            elif len(args) == 1:
                self._handlers[event](params)
            else:
                self._handlers[event](*params)

    def emit(self, event: str, *args, namespace: str = "/"):
        self._socket.emit(event, data=args, namespace=namespace)

    def disconnect(self):
        self._socket.disconnect()

    def connect(self, url: str, port: int, namespace: str):
        self._socket.connect(f'{url}:{port}',
                             transports=['websocket'],
                             namespaces=["/", namespace],
                             wait_timeout=self._timeout_s)

    def is_connected(self) -> bool:
        return self._socket.connected

    def register_handler(self, event: str, callback: Callable[[dict], None]):
        if event in self._handlers:
            raise Exception(f"Overwriting handler for event {event}")
        self._handlers[event] = callback

    def receive(self) -> SocketResponse:
        message_event = MessageEvent()

        with self._lock:
            self._receive_events.add(message_event)

        try:
            resp: SocketResponse = message_event.wait(self._timeout_s)
            return resp
        finally:
            self._receive_events.remove(message_event)
