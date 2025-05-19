
import logging
from pathlib import Path
import traceback

import socketio

from apps.cli_client.commands.cancel_active_job import CancelActiveJobCommand
from apps.cli_client.commands.change_state import ChangeStateCommand
from apps.cli_client.commands.pause_active_job import PauseActiveJobCommand

from apps.cli_client.services.config_service import ConfigService

from apps.cli_client.services.job_service import JobService
from core.http_api.http_api import HttpAPI
from apps.cli_client.services.module_service import ModuleService
from apps.cli_client.services.pip_service import PipService
from apps.cli_client.states.unconnected import UnconnectedState
from core.socket_api.api_client import APIClient
from core.cli_state_machine.state_machine_control import StateMachineControl
from utils.injector import Injector, get_injector
import asyncio


logging.basicConfig(level=logging.INFO)
CONFIG_FILE = './apps/cli_client/client_cfg.json'
BASE_URL = 'http://localhost:5000'


class CustomAPIClient(APIClient):
    def __init__(self,
                 server: str, port: int,
                 timeout_s: float,
                 sm: StateMachineControl) -> None:
        super().__init__(server, port, timeout_s)
        self._sm = sm

    def on_activation_requested(self) -> None:
        self._sm.dispatch(ChangeStateCommand(True))

    def on_release_requested(self) -> None:
        self._sm.dispatch(ChangeStateCommand(False))

    def on_drop_active_job_requested(self) -> None:
        self._sm.dispatch(PauseActiveJobCommand())

    def on_cancel_active_job_requested(self) -> None:
        self._sm.dispatch(CancelActiveJobCommand())


async def initialize_services(injector: Injector):
    config_service = ConfigService(Path(CONFIG_FILE))
    module_service = ModuleService(config_service().root)
    api_service = HttpAPI(BASE_URL)
    pip_service = PipService()
    job_service = JobService(config_service().root)

    injector.bind(ConfigService, to=config_service)
    injector.bind(ModuleService, to=module_service)
    injector.bind(HttpAPI, to=api_service)
    injector.bind(PipService, to=pip_service)
    injector.bind(JobService, to=job_service)

    await module_service.load_all_modules()


async def main():

    injector = get_injector()

    await initialize_services(injector)

    sm = StateMachineControl()

    cfg = injector.resolve(ConfigService).config

    while True:
        with CustomAPIClient(cfg.server, cfg.port, 5, sm) as client:
            injector.bind(APIClient, to=client, overwrite=True)
            try:
                await sm.run(UnconnectedState.instance())
                break
            except socketio.exceptions.TimeoutError or TimeoutError:
                logging.error("Connection timedout!")
            except Exception as e:
                logging.error("Error in state-machine controller: "
                              f"{e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    asyncio.run(main())
