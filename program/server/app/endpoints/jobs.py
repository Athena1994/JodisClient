

from dataclasses import dataclass
from flask import Blueprint, request
from injector import inject

from program.server.server import Server


jobs_pb = Blueprint('jobs_pb', __name__)


@dataclass
class Job:
    id: int
    state: str
    sub_state: str
    client_id: int


@jobs_pb.route('/jobs', methods=['GET'])
@inject
def get_jobs(server: Server):
    all = len(request.args) == 0
    assigned = all or 'assigned' in request.args
    unassigned = all or 'unassigned' in request.args
    finished = all or 'finished' in request.args

    jobs = []
    for j in server.get_jobs(include_unassigned=unassigned,
                             include_assigned=assigned,
                             include_finished=finished):
        print(j.id)
        print(j.states[-1].state.value)
        print(j.states[-1].sub_state.value)
        print(j.client.id if j.client is not None else -1)
        jobs.append(Job(j.id,
                        j.states[-1].state.value,
                        j.states[-1].sub_state.value,
                        j.client.id if j.client is not None else -1))


    return jobs
