

import enum
import logging
from pathlib import Path
import traceback

from cli_client.services.config_service import ConfigService
from cli_client.services.module_service import ModuleService

from jodiscore.dataobjects.job import JobDO
from jodiscore.client.http_api.http_api import HttpAPI
from jodiscore.client.modules.module_control import ClientModuleControl
from jodiscore.exceptions.invalid_state_error import InvalidStateError
from jodisutils.architecture.cache import Cache
from jodisutils.architecture.injector import inject

from jodiscore.client.modules.module_job_result import ModuleJobResult


class CacheKeys(enum.Enum):
    CURRENT_JOB = 'current-job'


class JobService:
    def __init__(self, root_dir: Path):
        if not root_dir.exists():
            raise FileNotFoundError(f"Root path {root_dir} does not exist.")

        self._working_dir = root_dir
        self._cache = Cache(root_dir, "./cache")
        self._control: ClientModuleControl = None

    # --- properties ---
    @property
    def has_running_job(self) -> bool:
        return self._control and self._control.is_busy

    # --- public methods ---
    @inject
    async def run_next(self, api: HttpAPI, cfg: ConfigService) -> bool:
        """
        Retrieve and start/continue the next job in the queue.
        """

        if self.has_running_job:
            raise InvalidStateError("Already has running job.")

        client_id = cfg().client_id

        if client_id == -1:
            raise InvalidStateError("Client not registered.")

        assigned_jobs = sorted(api.jobs.list_assigned_jobs(client_id),
                               key=lambda x: x.rank)

        if len(assigned_jobs) == 0:
            logging.info("No assigned jobs available.")
            return False

        next_job = assigned_jobs[0]

        if next_job.sub_state == 'FAILED':
            logging.info(f"Last job execution (id: {next_job.id}) failed.")
        elif next_job.sub_state == 'ABORTED':
            logging.info(f"Last job execution (id: {next_job.id}) was aborted.")

        if next_job.sub_state != 'RUNNING':
            try:
                api.jobs.start_job(next_job.id)
            except Exception as e:
                logging.error(f"Failed to start job {next_job.id}: {e}")
                raise e

        try:
            await self._prepare_job(job=next_job)
            result = await self._control.start_and_run(next_job, self._cache)

        except Exception as e:
            result = ModuleJobResult(error=e, traceback=traceback.format_exc())

        if result.error:
            logging.error(f"Job execution (id: {next_job.id}) failed: "
                          f"{result.error_str}")
            api.jobs.update_execution_state(next_job.id, failed=True)
        elif result.aborted:
            logging.info(f"Job execution (id: {next_job.id}) aborted.")
            api.jobs.update_execution_state(next_job.id, aborted=True)
        elif result.result is None:
            logging.warning(f"Job execution (id: {next_job.id}) finished "
                            f"successfully without valid result.")
            api.jobs.finish_job(next_job.id, None)
        else:
            logging.info(f"Job execution (id: {next_job.id}) finished "
                         f"successfully.")

            api.jobs.finish_job(next_job.id, result.json_result)

        return True

    # --- private methods ---

    def _at_working_dir(self, path: Path):
        """
        Join the working directory with the given path.
        :param path: The path to join.
        :return: The joined path.
        """
        return self._working_dir / path

    @inject
    async def _prepare_job(self, ms: ModuleService, job: JobDO):

        # assure module compatibility
        if not (ms.has_module(job.module_name)
                and ms.is_module_compatible(job.module_name, job.module_id)):
            logging.info(f"Module {job.module_name} not found or not "
                         "compatible.")
            await ms.install_or_update(job.module_id)
        else:
            logging.info(f"Module {job.module_name} found and compatible.")

        # prepare module
        module = ms.get_module_by_name(job.module_name)
        if not module.is_ready:
            raise InvalidStateError(f"Module {job.module_name} not ready.")
        self._control = module.control
        logging.info(f"Module {job.module_name} control prepared.")
