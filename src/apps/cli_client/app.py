
import inspect
import logging

# from apps.cli_client.commands.cancel_active_job import CancelActiveJobCommand
# from apps.cli_client.commands.change_state import ChangeStateCommand
# from apps.cli_client.commands.pause_active_job import PauseActiveJobCommand

from apps.cli_client.services.config_service import ConfigService

from core.api.api_client import APIClient
from core.cli_state_machine.base_state import BaseState, CliCommand
from core.cli_state_machine.state_machine_control import StateMachineControl
from utils.cli.cli_command import CLICommand


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



def main():

    config_service = ConfigService(CONFIG_FILE)

    context = {
        'abc': 1,
    }

    sm = StateMachineControl(context)

    sm.run(TestState())

    # while True:
    #     init_context = {
    #         'cfg': cfg,
    #     }
    #     with CustomAPIClient(cfg.server, cfg.port, 5, sm) as client:
    #         try:
    #             sm.run(init_context, UnconnectedState(client))
    #             break
    #         except socketio.exceptions.TimeoutError or TimeoutError:
    #             logging.error("Connection timedout!")
    #         except Exception as e:
    #             logging.error(f"Error: {e}")


if __name__ == '__main__':
    main()
