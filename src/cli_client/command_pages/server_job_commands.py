

from jodisutils.cli.cli_state_machine.cli_command_decorator import CliCommand
from jodisutils.cli.cli_state_machine.cli_command_page import CLICommandPage
from jodiscore.client.http_api.http_api import HttpAPI
from jodisutils.architecture.injector import inject


class ServerJobCommands(CLICommandPage):

    @CliCommand('unassigned-jobs')
    @inject
    def _list_jobs(self, api: HttpAPI):
        jobs = api.jobs.list_unassigned_jobs()
        print(f"Unassigned Jobs ({len(jobs)}):")
        for job in jobs:
            print(f"\t{job.name} (ID: {job.id})")

    @CliCommand('assigned-jobs', inject_context=True)
    @inject
    def _list_assigned_jobs(self, context: dict,
                            api: HttpAPI, client_id: int = None):
        if client_id is None:
            if 'id' not in context:
                print("Client ID not given and not found in context.")
                return
            client_id = context['id']

        jobs = api.jobs.list_assigned_jobs(client_id)
        print(f"Assigned Jobs (client_id: {client_id}, #{len(jobs)}):")
        for job in jobs:
            print(f"\t{job.name} (ID: {job.id}, state: {job.sub_state})")

    @CliCommand('assign-job', inject_context=True)
    @inject
    def _assign_job(self, api: HttpAPI, context: dict,
                    job_id: int, client_id: int = None):
        if client_id is None:
            if 'id' not in context:
                print("Client ID not given and not found in context.")
                return
            client_id = context['id']

        try:
            api.jobs.assign_job(job_id, client_id)
            print(f"Assigned job {job_id} to client.")
        except Exception as e:
            print(f"Failed to assign job {job_id}. ({e})")

    @CliCommand('unassign-job', inject_context=True)
    @inject
    def _unassign_job(self, api: HttpAPI, context: dict,
                      job_id: int):
        try:
            api.jobs.unassign_job(job_id)
            print(f"Unassigned job {job_id}.")
        except Exception as e:
            print(f"Failed to unassign job {job_id}: {e}")

    @CliCommand('claim-next-job', inject_context=True)
    @inject
    def _next_job(self, api: HttpAPI, context: dict):
        if 'id' not in context:
            print("Client ID not found in context.")
            return
        client_id = context['id']

        unassigned_jobs = api.jobs.list_unassigned_jobs()

        if len(unassigned_jobs) == 0:
            print("No unassigned jobs available.")
            return

        job_id = unassigned_jobs[0].id
        try:
            api.jobs.assign_job(job_id, client_id)
            print(f"Job {job_id} assigned to client.")
        except Exception as e:
            print(f"Failed to assign job {job_id}: {e}")

    @CliCommand('set-next-job-running', inject_context=True)
    @inject
    def _start_next_job(self, api: HttpAPI, context: dict):

        if 'id' not in context:
            print("Client ID not found in context.")
            return
        client_id = context['id']

        assigned_jobs = api.jobs.list_assigned_jobs(client_id)

        if any(job.sub_state == 'RUNNING' for job in assigned_jobs):
            print("Client already has a running job.")
            return

        if len(assigned_jobs) == 0:
            print("No assigned jobs available.")
            return

        job_id = sorted(assigned_jobs, key=lambda j: j.rank)[0].id
        try:
            api.jobs.start_job(job_id)
            print(f"Job {job_id} set to RUNNING.")
        except Exception as e:
            print(f"Failed to start job {job_id}: {e}")
            return
