
import logging

import socketio

# from apps.cli_client.commands.cancel_active_job import CancelActiveJobCommand
# from apps.cli_client.commands.change_state import ChangeStateCommand
# from apps.cli_client.commands.pause_active_job import PauseActiveJobCommand

from apps.cli_client.services.config_service import ConfigService

#from apps.cli_client.states.unconnected import UnconnectedState
from core.api.api_client import APIClient
from core.cli_state_machine.state_machine_control import StateMachineControl
from utils.injector import get_injector, inject


logging.basicConfig(level=logging.INFO)
CONFIG_FILE = 'client_cfg.json'


class CustomAPIClient(APIClient):
    def __init__(self,
                 server: str, port: int,
                 timeout_s: float,
                 sm: StateMachineControl) -> None:
        super().__init__(server, port, timeout_s)
        self._sm = sm

    # def on_activation_requested(self) -> None:
    #     self._sm.dispatch_command(ChangeStateCommand(True))

    # def on_release_requested(self) -> None:
    #     self._sm.dispatch_command(ChangeStateCommand(False))

    # def on_drop_active_job_requested(self) -> None:
    #     self._sm.dispatch_command(PauseActiveJobCommand())

    # def on_cancel_active_job_requested(self) -> None:
    #     self._sm.dispatch_command(CancelActiveJobCommand())


class Test:
    def test(self):
        print("Test class")


@inject
def test(b: int, a: Test,  c=3):
    a.test()
    print(b, c)


def main():

    config_service = ConfigService(CONFIG_FILE)

    cfg = config_service.config

    sm = StateMachineControl()

    get_injector().register(Test())

    test(2)

    return

    # while True:
    #     with CustomAPIClient(cfg.server, cfg.port, 5, sm) as client:
    #         try:
    #             sm.run(UnconnectedState(client))
    #             break
    #         except socketio.exceptions.TimeoutError or TimeoutError:
    #             logging.error("Connection timedout!")
    #         except Exception as e:
    #             logging.error(f"Error: {e}")


if __name__ == '__main__':
    main()
