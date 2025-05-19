

from typing import List
from core.http_api.api_objects import JobDO
from core.http_api.http_api_base import HttpApiBase


class HttpJobsAPI(HttpApiBase):
    """
    HTTP API for jobs.
    """
    def __init__(self, base_url: str):
        super().__init__(base_url)

    def list_unassigned_jobs(self) -> List[JobDO]:
        """
        List unassigned jobs.
        :return: List of unassigned jobs.
        """
        try:
            response = self.get("/jobs/unassigned")
            return [JobDO(**job) for job in response]
        except Exception as e:
            raise ValueError(f"Server returned {e}")

    def list_assigned_jobs(self, id: int) -> List[JobDO]:
        """
        List assigned jobs.
        :return: List of assigned jobs.
        """
        try:
            response = self.get(f"/client/{id}/jobs")
            return [JobDO(**job) for job in response]
        except Exception as e:
            raise ValueError(f"Server returned {e}")

    def assign_job(self, job_id: int, client_id: int) -> bool:
        """
        Assign a job to a client.
        :param job_id: Job ID.
        :param client_id: Client ID.
        """
        try:
            self.post(f"/client/{client_id}/jobs/{job_id}")
            return True
        except Exception as e:
            raise ValueError(f"Server returned {e}")

    def unassign_job(self, job_id: int) -> bool:
        """
        Assign a job to a client.
        :param job_id: Job ID.
        :param client_id: Client ID.
        """
        try:
            self.post(f"/jobs/unassign/{job_id}")
            return True
        except Exception as e:
            raise ValueError(f"Server returned {e}")

    def start_job(self, job_id: int) -> bool:
        """
        Start a job.
        :param job_id: Job ID.
        """
        try:
            self.post(f"/jobs/start/{job_id}")
            return True
        except Exception as e:
            raise ValueError(f"Server returned {e}")

    def validate(self, module_id: str, job_config: dict) -> bool:
        """
        Validate a job config.
        :param module_id: Module ID.
        :param job_config: Job config.
        """
        try:
            result = self.post(
                "/job/validate",
                query_params={'module-id': module_id},
                json=job_config
            )
            return result.get("valid", False)
        except Exception as e:
            raise ValueError(f"Server returned {e}")

    def create(self, module_id: str, job_config: dict, name: str) -> bool:
        """
        Create a job.
        :param module_id: Module ID.
        :param job_config: Job config.
        :param name: Job name.
        """
        try:
            return self.post(f"job/{module_id}",
                             query_params={'name': name}, json=job_config)
        except Exception as e:
            raise ValueError(f"Server returned {e}")
