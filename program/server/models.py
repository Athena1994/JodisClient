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
    name: Mapped[str] = mapped_column("Name", String(64), nullable=True)
    description: Mapped[str] = mapped_column(
        "Description", String(256), nullable=True)

    states: Mapped[List["JobStatus"]] = relationship(
        back_populates='job', cascade='all, delete-orphan',)
    client: Mapped["Client"] = relationship(back_populates='job')


class JobStatus(Base):
    class SubState(enum.Enum):
        CREATED = 'CREATED'
        RETURNED = 'RETURNED'

        PENDING = 'PENDING'
        IN_PROGRESS = 'ASSIGNED'
        PAUSED = 'PAUSED'

        FAILED = 'FAILED'
        FINISHED = 'FINISHED'
        ABORTED = 'ABORTED'

    class State(enum.Enum):
        UNASSIGNED = 'UNASSIGNED'
        ASSIGNED = 'ASSIGNED'
        FINISHED = 'FINISHED'

    __tablename__ = 'JobStatus'

    id: Mapped[int] = mapped_column("Id", primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(
        "JobId", ForeignKey('Job.Id', ondelete='CASCADE'))

    state: Mapped[State] = mapped_column("State")
    sub_state: Mapped[SubState] = mapped_column("SubState")
    creation_timestamp: Mapped[datetime] = mapped_column(
        'Timestamp', default=func.current_timestamp())

    job: Mapped["Job"] = relationship(back_populates='states')


class Client(Base):
    __tablename__ = 'Client'

    id: Mapped[int] = mapped_column("Id", primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(
        "JobId", ForeignKey('Job.Id', ondelete='SET NULL'), nullable=True)

    name: Mapped[str] = mapped_column("Name", String(64), nullable=True)

    job: Mapped["Job"] = relationship(back_populates='client')
    connection_states: Mapped[List["ClientConnectionState"]] \
        = relationship(back_populates='client',
                       cascade='all, delete-orphan',
                       passive_deletes=True)


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


ConnectionState = ClientConnectionState.State


def create_initial_job_status(mapper,
                              connection,
                              target: Job):
    status = JobStatus(state=JobStatus.State.UNASSIGNED,
                       sub_state=JobStatus.SubState.CREATED)
    target.states.append(status)


def create_initial_connection(mapper,
                              connection,
                              target: Client):
    status = ClientConnectionState(state=ConnectionState.DISCONNECTED)
    target.connection_states.append(status)


event.listen(Job, 'after_insert', create_initial_job_status)
event.listen(Client, 'after_insert', create_initial_connection)
