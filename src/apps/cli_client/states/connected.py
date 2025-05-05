
from socket import gethostname
from apps.cli_client.states.active import ActiveState
from apps.cli_client.states.suspended import SuspendedState
from apps.cli_client.states.unconnected import UnconnectedState
from core.api.api_client import APIClient
from core.cli_state_machine.base_state import BaseState
from core.cli_state_machine.cli_command_handler import CLICommandHandler


class ConnectedState(BaseState):
    def __init__(self, client: APIClient):
        super().__init__()
        self._client = client

        self.add_handler('disconnect', self._disconnect)
        self.add_handler('register', self._register,
                         [CLICommandHandler.Parameter('name', str, gethostname()),
                          CLICommandHandler.Parameter('save', bool, 'False')])
        self.add_handler('list', self._get_clients)
        self.add_handler('claim', self._claim,
                         [CLICommandHandler.Parameter(
                            'id', int,
                            default_expr='context["cfg"].client_id')])

    def prompt_prefix(self, _) -> str:
        return 'connected'

    # --- properties ---

    @property
    def client(self) -> APIClient:
        return self._client

    # --- handlers -----

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
