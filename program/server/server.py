from dataclasses import dataclass
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from .models import Base, Client, Job, JobStatus
from utils.config_utils import assert_fields_in_dict


class Server:

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

        self._session = Session(self._engine)

    def get_jobs(self):
        return self._session.query(Job).all()

    def add_client(self, name: str):
        client = Client(name=name)
        self._session.add(client)
        self._session.commit()
        return client.id

    def add_job(self, job_config: dict):
        self._session.add(Job(configuration=job_config))
        self._session.commit()

    def assign_job(self, job_id: int):
        job = self._session.query(Job).filter(Job.id == job_id).first()
        if job is None:
            raise ValueError(f"Job with id {job_id} not found")

        job.states.append(JobStatus(status=JobStatus.State.STARTED))
        self._session.commit()

    def create_tables(self):
        Base.metadata.drop_all(self._engine)
        Base.metadata.create_all(self._engine)
