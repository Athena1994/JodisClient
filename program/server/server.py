from dataclasses import dataclass
import logging
from sqlalchemy import and_, create_engine, func, or_, select, tuple_
from sqlalchemy.orm import Session

from .models import (Base, Client, ClientConnectionState, Job, JobStatus)

from utils.config_utils import assert_fields_in_dict

JState = JobStatus.State
CState = ClientConnectionState.State


class Server:

    class IndexValueError(ValueError):
        pass

    class StateError(ValueError):
        pass

    @dataclass
    class Config:
        sql_user: str
        sql_pw: str
        sql_server: str
        sql_db: str

        @staticmethod
        def from_dict(cfg: dict):
            assert_fields_in_dict(cfg, ['user', 'password', 'host', 'db'])
            return Server.Config(cfg['user'], cfg['password'],
                                 cfg['host'], cfg['db'])

        @staticmethod
        def get_test_config():
            return Server.Config("", "", "", "")

    def __init__(self, cfg: Config):
        if len(cfg.sql_user) == 0:
            credentials = '/'
        else:
            credentials = f"{cfg.sql_user}:{cfg.sql_pw}@"

        if len(cfg.sql_server) == 0:
            target = ':memory:'
        else:
            target = f"{cfg.sql_server}/{cfg.sql_db}"

        connect_query = f'mysql+pymysql://{credentials}{target}'
        self._engine = create_engine(connect_query)

        self._socket_to_client = {}
        self._client_to_socket = {}

    def create_session(self) -> Session:
        return Session(self._engine)

    def delete_job(self, session: Session, job_id: int):
        logging.info(f"Deleting job with id {job_id}")
        job = session.execute(select(Job).where(Job.id == job_id)).scalar()
        if job is None:
            raise Server.IndexValueError(f"Job with id {job_id} not found")

        if job.states[-1].state == JState.ASSIGNED:
            raise Server.StateError("Assigned jobs cannot be deleted.")

        session.delete(job)
        session.commit()

    def get_jobs(self,
                 session: Session,
                 include_unassigned: bool,
                 include_finished: bool,
                 include_assigned: bool):
        most_recent_state_stmt = (
            select(JobStatus.job_id, JobStatus.state)
            .where(
                tuple_(JobStatus.job_id, JobStatus.creation_timestamp)
                .in_(
                    select(JobStatus.job_id,
                           func.max(JobStatus.creation_timestamp))
                    .group_by(JobStatus.job_id)))
        )

        jobs_stmt = (
            select(Job)
            .where(
                or_(
                    and_(include_unassigned,
                         tuple_(Job.id, JState.UNASSIGNED.value)
                         .in_(most_recent_state_stmt)),
                    and_(include_assigned,
                         tuple_(Job.id, JState.ASSIGNED.value)
                         .in_(most_recent_state_stmt)),
                    and_(include_finished,
                         tuple_(Job.id, JState.FINISHED.value)
                         .in_(most_recent_state_stmt))
                )
            )
        )
        result = session.execute(jobs_stmt).scalars()
        return result

    def get_job(self,
                session: Session,
                job_id: int):
        return session.execute(
            select(Job).where(Job.id == job_id)
        ).scalar()

    def add_job(self, job_config: dict, name: str, desc: str) -> int:
        with Session(self._engine) as session:
            job = Job(configuration=job_config,
                      name=name,
                      description=desc)
            session.add(job)
            session.commit()
            return job.id

    def assign_job(self, job_id: int):
        with Session(self._engine) as session:
            job = session.query(Job).filter(Job.id == job_id).first()
            if job is None:
                raise ValueError(f"Job with id {job_id} not found")

            job.states.append(JobStatus(s=JobStatus.State.ASSIGNED,
                                        sub_state=JobStatus.SubState.PENDING))
            session.commit()

    def add_client(self, name: str):
        client = Client(name=name)
        with Session(self._engine) as session:
            session.add(client)
            session.commit()
            return client.id

    def get_all_clients(self, session):
        result = session.query(Client).all()
        return result

    def register_socket(self, socket_id: int, client_id: int):
        if socket_id in self._socket_to_client:
            raise ValueError(f"Socket {socket_id} already registered")

        with Session(self._engine) as session:

            client = session.query(Client)\
                .filter(Client.id == client_id)\
                .first()
            if client is None:
                raise ValueError(f"Client with id {client_id} not found")

            self._socket_to_client[socket_id] = client_id
            self._client_to_socket[client_id] = socket_id

            client.connection_states.append(
                ClientConnectionState(state=CState.CONNECTED,
                                      message='Connected'))
            session.commit()

    def deregister_socket(self, socket_id: int) -> int:
        if socket_id not in self._socket_to_client:
            return None

        client_id = self._socket_to_client[socket_id]

        del self._socket_to_client[socket_id]
        del self._client_to_socket[client_id]

        with Session(self._engine) as session:
            client = session.query(Client)\
                .filter(Client.id == client_id)\
                .first()
            client.connection_states.append(
                ClientConnectionState(
                    state=ClientConnectionState.State.DISCONNECTED,
                    message='Disonnected'))
            session.commit()

        return client_id

    def create_tables(self):
        Base.metadata.drop_all(self._engine)
        Base.metadata.create_all(self._engine)
