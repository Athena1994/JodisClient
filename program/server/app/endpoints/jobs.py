

from dataclasses import dataclass
import logging
from flask import Blueprint, request
from injector import inject

from program.config_loader import ConfigLoader
from program.server.server import Server


jobs_pb = Blueprint('jobs_pb', __name__)


@dataclass
class Job:
    id: int
    state: str
    sub_state: str
    client_id: int
    config: dict
    name: str
    description: str

    @staticmethod
    def from_db(job):
        return Job(job.id,
                   job.states[-1].state.value,
                   job.states[-1].sub_state.value,
                   job.client.id if job.client is not None else -1,
                   job.configuration,
                   job.name,
                   job.description)


@jobs_pb.route('/jobs', methods=['GET'])
@inject
def get_jobs(server: Server):
    all = len(request.args) == 0
    assigned = all or 'assigned' in request.args
    unassigned = all or 'unassigned' in request.args
    finished = all or 'finished' in request.args

    jobs = []
    with server.create_session() as session:
        for j in server.get_jobs(session,
                                 include_unassigned=unassigned,
                                 include_assigned=assigned,
                                 include_finished=finished):
            jobs.append(Job.from_db(j))

    return jobs


@jobs_pb.route('/job/validate', methods=['POST'])
@inject
def validate_config(server: Server):
    try:
        ConfigLoader(request.json)
        return {'valid': True}
    except ValueError as e:
        return {'valid': False, 'message': str(e)}


@jobs_pb.route('/jobs/delete', methods=['POST'])
@inject
def delete_jobs(server: Server):
    if 'ids' not in request.json:
        return {'error': 'Ids not provided'}, 400

    ids = request.json['ids']

    if not all(isinstance(i, int) for i in ids):
        return {'error': 'Ids should be integers'}, 400

    deleted_ids = []
    with server.create_session() as session:
        for id in ids:
            try:
                server.delete_job(session, id)
                deleted_ids.append(id)
            except server.StateError or server.IndexValueError as e:
                logging.warning(f'Failed to delete job {id}: {str(e)}')
            except Exception as e:
                logging.error(f'Failed to delete job {id}: {str(e)}')

    return {'deletedIds': deleted_ids}, 200


@jobs_pb.route('/job', methods=['POST'])
@inject
def create_job(server: Server):
    if 'name' not in request.json:
        logging.warning('Name not provided')
        return {'error': 'Name not provided'}, 400
    if 'config' not in request.json:
        logging.warning('config not provided')
        return {'error': 'Config not provided'}, 400
    if 'description' not in request.json:
        logging.warning('description not provided')
        return {'error': 'Description not provided'}, 400

    try:
        ConfigLoader(request.json['config'])
    except ValueError as e:
        logging.warning('Provided config is invalid')
        return {'error': f"Invalid config: {str(e)}"}, 400

    job_id = server.add_job(request.json['config'],
                            request.json['name'],
                            request.json['description'])

    with server.create_session() as session:
        job = Job.from_db(server.get_job(session, job_id))
        logging.info(f'Created job {job}')
        return [job], 201
