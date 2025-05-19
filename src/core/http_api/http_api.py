

from core.http_api.jobs import HttpJobsAPI
from core.http_api.modules import HttpModulesAPI


class HttpAPI:
    def __init__(self, base_url: str):
        self._modules_api = HttpModulesAPI(base_url)
        self._jobs_api = HttpJobsAPI(base_url)

    @property
    def modules(self) -> HttpModulesAPI:
        return self._modules_api

    @property
    def jobs(self) -> HttpJobsAPI:
        return self._jobs_api
