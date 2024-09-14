
from dataclasses import dataclass
import dataclasses
import json
import logging
import os
from socket import gethostname

import socketio
import socketio.exceptions

from api_client import APIClient
from utils.cli_state_machine import (BaseState, CLIStateMachine, Command,
                                     Parameter)


logging.basicConfig(level=logging.INFO)
CONFIG_FILE = 'client_cfg.json'


@dataclass
class Config:
    server: str
    port: int

    client_id: int

    @staticmethod
    def from_dict(cfg: dict):

        return Config(cfg['server'], cfg['port'], cfg['client_id'])

    def save(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(dataclasses.asdict(self), f)

    @staticmethod
    def load():
        with open(CONFIG_FILE, 'r') as f:
            return Config.from_dict(json.load(f))


class UnconnectedState(BaseState):
    def __init__(self, client: APIClient):
        super().__init__()
        self.client = client
        self.add_handler('connect', self._connect)

    def _connect(self, params: dict, context: dict) -> BaseState:

        if self.client.connect():
            return ConnectedState(self.client)
        else:
            return self

    def prompt_prefix(self, context: dict) -> str:
        return 'unconnected'


class ConnectedState(BaseState):
    def __init__(self, client: APIClient):
        super().__init__()
        self.client = client

        self.add_handler('disconnect', self._disconnect)
        self.add_handler('register', self._register,
                         [Parameter('name', gethostname()),
                          Parameter('save', 'False')])
        self.add_handler('list', self._get_clients)
        self.add_handler('claim', self._claim,
                         [Parameter('id',
                                    default_expr='context["cfg"].client_id')])

    def _get_clients(self, _, context: dict) -> 'BaseState':
        clients = self.client.get_client_list()
        for client in clients:
            print(f"{client['id']}: {client['name']}")
        return self

    def _register(self, params: dict, context: dict) -> 'BaseState':
        cfg: Config = context['cfg']
        cfg.client_id = self.client.register(params['name'])
        if params['save']:
            cfg.save()
        return self

    def _disconnect(self, _, __) -> 'BaseState':
        self.client.disconnect()
        return UnconnectedState(self.client)

    def _claim(self, params: dict, _) -> 'BaseState':
        result = self.client.claim_client(int(params['id']))
        if result:
            if result['state'] == 'ACTIVE':
                return ActiveState(self.client,
                                   result['name'], result['id'])
            else:
                return SuspendedState(self.client,
                                      result['name'], result['id'])
        else:
            print('Failed to claim client')
            return self

    def prompt_prefix(self, _) -> str:
        return 'connected'


class ClaimedState(ConnectedState):
    def __init__(self,
                 client: APIClient,
                 client_name: str,
                 client_id: int):
        super().__init__(client)
        self._client_name = client_name
        self._client_id = client_id
        self.add_handler('drop', self._drop_claim)

    def _drop_claim(self, _, __) -> 'BaseState':
        self.client.drop_claim()
        return ConnectedState(self.client)

    def prompt_prefix(self, _) -> str:
        return (
            f'connected ({self._client_id}: {self._client_name}, '
            f'{self._client_state})'
        )


class ActiveState(ClaimedState):
    def __init__(self,
                 client: APIClient,
                 client_name: str,
                 client_id: int):
        super().__init__(client, client_name, client_id)

        self.add_handler('deactivate', self._release_active)

        self.add_handler('start_next', self._start_next)

    def _start_next(self, _, __) -> 'BaseState':
        self.client.claim_next_job()
        return self

    def _release_active(self, _, __) -> 'BaseState':
        new_state = self.client.release_active_state()
        if new_state:
            print(f"Entered suspended state ({new_state})")
            return SuspendedState(self.client,
                                  self._client_name, self._client_id)
        else:
            print("Failed to release active state")
            return self

    def prompt_prefix(self, _) -> str:
        return (
            f'connected ({self._client_id}: {self._client_name}, active)'
        )


class SuspendedState(ClaimedState):
    def __init__(self,
                 client: APIClient,
                 client_name: str,
                 client_id: int):
        super().__init__(client, client_name, client_id)

        self.add_handler('activate', self._claim_active)

    def _claim_active(self, _, __) -> 'BaseState':
        new_state = self.client.claim_active_state()
        if new_state:
            print(f"Claimed active state ({new_state})")
            return ActiveState(self.client,
                               self._client_name, self._client_id)
        else:
            print("Failed to claim active state")
            return self

    def prompt_prefix(self, _) -> str:
        return (
            f'connected ({self._client_id}: {self._client_name}, suspended)'
        )


class ChangeStateCommand(Command[ClaimedState]):
    def __init__(self, activate: bool) -> None:
        super().__init__()
        self._activate = activate

    def run(self, state: ClaimedState) -> BaseState:
        if self._activate and isinstance(state, SuspendedState):
            return state._claim_active(None, None)
        elif not self._activate and isinstance(state, ActiveState):
            return state._release_active(None, None)
        else:
            return state


class CustomAPIClient(APIClient):
    def __init__(self,
                 server: str, port: int,
                 timeout_s: float,
                 sm: CLIStateMachine) -> None:
        super().__init__(server, port, timeout_s)
        self._sm = sm

    def on_activation_requested(self) -> None:
        self._sm.dispatch_command(ChangeStateCommand(True))

    def on_release_requested(self) -> None:
        self._sm.dispatch_command(ChangeStateCommand(False))


def main():
    if os.path.exists(CONFIG_FILE):
        cfg = Config.load()
    else:
        cfg = Config('localhost', 5000, -1)

    sm = CLIStateMachine()

    while True:
        init_context = {
            'cfg': cfg,
        }
        with CustomAPIClient(cfg.server, cfg.port, 5, sm) as client:
            try:
                sm.run(init_context, UnconnectedState(client))
                break
            except socketio.exceptions.TimeoutError or TimeoutError:
                logging.error("Connection timedout!")
            except Exception as e:
                logging.error(f"Error: {e}")


if __name__ == '__main__':
    main()
