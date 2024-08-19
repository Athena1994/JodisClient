import enum
from typing import List
from sqlalchemy import JSON, ForeignKey, String, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from datetime import datetime


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = 'Job'

    id: Mapped[int] = mapped_column("Id", primary_key=True, autoincrement=True)

    configuration: Mapped[JSON] = mapped_column('Configuration', type_=JSON)
    creation_timestamp: Mapped[datetime] = mapped_column(
        'CreationTimestamp', default=func.current_timestamp())

    states: Mapped[List["JobStatus"]] = relationship(back_populates='job')
    client: Mapped["Client"] = relationship(back_populates='job')


class JobStatus(Base):
    class State(enum.Enum):
        CREATED = 'CREATED'
        IN_PROGRESS = 'ASSIGNED'
        STARTED = 'STARTED'
        PAUSED = 'PAUSED'
        FAILED = 'FAILED'
        FINISHED = 'FINISHED'

    __tablename__ = 'JobStatus'

    id: Mapped[int] = mapped_column("Id", primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(
        "JobId", ForeignKey('Job.Id', ondelete='CASCADE'))

    status: Mapped[State] = mapped_column("State")
    creation_timestamp: Mapped[datetime] = mapped_column(
        'Timestamp', default=func.current_timestamp())

    job: Mapped["Job"] = relationship(back_populates='states')


class Client(Base):
    __tablename__ = 'Client'

    id: Mapped[int] = mapped_column("Id", primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column("JobId", ForeignKey('Job.Id'),
                                        nullable=True)

    name: Mapped[str] = mapped_column("Name", String(64), nullable=True)

    job: Mapped["Job"] = relationship(back_populates='client')
    connection_states: Mapped[List["ClientConnectionState"]] \
        = relationship(back_populates='client')


class ClientConnectionState(Base):
    class State(enum.Enum):
        CONNECTED = 'CONNECTED'
        DISCONNECTED = 'DISCONNECTED'

    __tablename__ = 'ClientConnectionState'

    id: Mapped[int] = mapped_column("Id", primary_key=True, autoincrement=True)
    client_id: Mapped[int] = mapped_column(
        "ClientId", ForeignKey('Client.Id', ondelete='CASCADE'))

    state: Mapped[State] = mapped_column("State")
    message: Mapped[str] = mapped_column("Message", String(128), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        'Timestamp', default=func.current_timestamp())

    client: Mapped["Client"] = relationship(back_populates='connection_states')


def create_initial_job_status(mapper,
                              connection,
                              target: Job):
    status = JobStatus(status=JobStatus.State.CREATED)
    target.states.append(status)


def create_initial_connection(mapper,
                              connection,
                              target: Client):
    status = ClientConnectionState(state=ClientConnectionState.State.CONNECTED)
    target.connection_states.append(status)


event.listen(Job, 'after_insert', create_initial_job_status)
event.listen(Client, 'after_insert', create_initial_connection)
