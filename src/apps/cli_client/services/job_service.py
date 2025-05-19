

import enum
import logging
from pathlib import Path
import traceback
from apps.cli_client.services.config_service import ConfigService
from apps.cli_client.services.module_service import ModuleService
from core.http_api.api_objects import JobDO
from core.http_api.http_api import HttpAPI
from core.modules.module_control import ClientModuleControl
from errors.invalid_state_error import InvalidStateError
from utils.cache import Cache
from utils.injector import inject
from utils.web_ressource import WebRessource


class CacheKeys(enum.Enum):
    CURRENT_JOB = 'current-job'


class JobService:
    def __init__(self, root_dir: Path):
        if not root_dir.exists():
            raise FileNotFoundError(f"Root path {root_dir} does not exist.")

        self._working_dir = root_dir
        self._has_running = False
        self._cache = Cache(root_dir, "./cache")
        self._control: ClientModuleControl = None

    # --- properties ---
    @property
    def has_running_job(self) -> bool:
        return self._has_running

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

        if next_job.sub_state != 'RUNNING':
            try:
                api.jobs.start_job(next_job.id)
            except Exception as e:
                logging.error(f"Failed to start job {next_job.id}: {e}")
                raise e

        await self._prepare_job(job=next_job)

        self._has_running = True

        try:
            await self._control.start(next_job, self._cache)
        except Exception as e:
            logging.error(f"Failed to run job {next_job.id}: {e}\n"
                          f"{traceback.format_exc()}")

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

        # prepare cache
        cached_job = self._cache.get(CacheKeys.CURRENT_JOB, JobDO, False)
        if cached_job is None or cached_job.id != job.id:
            self._cache.clear()
            self._cache.set(CacheKeys.CURRENT_JOB, job, True)

        # prepare module
        module = ms.get_module_by_name(job.module_name)
        if not module.is_ready:
            raise InvalidStateError(f"Module {job.module_name} not ready.")
        self._control = module.control

        # download payload
        WebRessource()

