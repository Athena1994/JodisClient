
from dataclasses import dataclass
from enum import Enum
import logging
from queue import Queue
import threading
from typing import List
from core.data.data_provider import ChunkType
from core.qlearning.dqn_trainer import DQNTrainer
from core.qlearning.q_arbiter import ExplorationArbiter
from core.simulation.experience_provider import ExperienceProvider
from core.simulation.sample_provider import SampleProvider
from core.simulation.state_manager import StateManager
from program.experience_evaluator import ExperienceEvaluator
from program.trading_environment import TradingEnvironment
from utils import benchmark


class TrainingReporter:

    @dataclass
    class Update:
        class Type(Enum):
            TRAINING_DATA = 0,
            VALIDATION_DATA = 1
            NEW_EPOCH = 2

        type: Type
        data: dict

    @dataclass
    class State:
        @dataclass
        class Progress:
            ix: int
            av_time: float
            last_time: float

            def copy(self):
                return TrainingReporter.State.Progress(
                    self.ix, self.av_time, self.last_time
                )

        class Phase(Enum):
            TRAINING = 0,
            VALIDATION = 1

        epoch: Progress
        training: Progress
        validation: Progress

        validation_samples: List[DQNTrainer.ExperienceTuple]
        last_evaluation: dict

        phase: Phase

        def copy(self):
            return TrainingReporter.State(
                self.epoch.copy(),
                self.training.copy(),
                self.validation.copy(),
                (self.validation_samples or []).copy(),
                (self.last_evaluation or {}).copy(),
                self.phase
            )

    def __init__(self, evaluator: ExperienceEvaluator):
        self._state = TrainingReporter.State(
            TrainingReporter.State.Progress(0, 0, 0),
            TrainingReporter.State.Progress(0, 0, 0),
            TrainingReporter.State.Progress(0, 0, 0),
            None, None,
            TrainingReporter.State.Phase.TRAINING)

        self._lock = threading.Lock()

        self._evaluator = evaluator

        self._update_queue = Queue()
        self._update_thread = None

    def is_running(self):
        return (self._update_thread is not None
                and self._update_thread.is_alive())

    def stop(self, blocking: bool):
        if self._update_thread is None:
            return

        thread = self._update_thread
        self._update_thread = None

        if blocking:
            thread.join()

    def start_update_thread(self):
        if self._update_thread is not None and self._update_thread.is_alive():
            raise RuntimeError('Update thread already started')

        def update_thread():
            while self._update_thread is not None:
                update = self._update_queue.get()
                self._perform_update(update)

        self._update_thread = threading.Thread(target=update_thread)
        self._update_thread.start()

    def get_state(self):
        with self._lock:
            return self._state.copy()

    def add_training_update(self,
                            new_sample_cnt: int,
                            av_time: float):
        self._add_update(
            TrainingReporter.Update.Type.TRAINING_DATA,
            {'cnt': new_sample_cnt,
             'av_time': av_time}
        )

    def add_validation_update(self,
                              experiences: List[DQNTrainer.ExperienceTuple],
                              av_time: float):
        self._add_update(
            TrainingReporter.Update.Type.VALIDATION_DATA,
            {'experiences': experiences,
             'av_time': av_time}
        )

    def add_epoch_update(self, av_time: float, last_time: float):
        self._add_update(
            TrainingReporter.Update.Type.NEW_EPOCH,
            {'av_time': av_time,
             'last_time': last_time}
        )

    def _add_update(self, type: Update.Type, data: dict):
        if self._update_thread is not None:
            self._update_queue.put(TrainingReporter.Update(type, data))
        else:
            logging.warning('Update thread stopping or not running. '
                            'Ignoring update.')

    def _perform_update(self, update):
        with self._lock:
            if update.type == TrainingReporter.Update.Type.TRAINING_DATA:
                if self._state.phase != TrainingReporter.State.Phase.TRAINING:
                    self._state.phase = TrainingReporter.State.Phase.TRAINING
                self._state.training.ix += update.data['cnt']
                self._state.training.av_time = update.data['av_time']

            elif update.type == TrainingReporter.Update.Type.VALIDATION_DATA:
                if self._state.phase != TrainingReporter.State.Phase.VALIDATION:
                    self._state.phase = TrainingReporter.State.Phase.VALIDATION
                    self._state.validation_samples = []
                lst = update.data['experiences']
                self._state.validation_samples.extend(lst)
                self._state.validation.ix += len(lst)
                self._state.validation.av_time = update.data['av_time']

                self._state.last_evaluation = self._evaluator.evaluate(
                    self._state.validation_samples
                )

            elif update.type == TrainingReporter.Update.Type.NEW_EPOCH:
                self._state.epoch.ix += 1
                self._state.epoch.av_time = update.data['av_time']
                self._state.epoch.last_time = update.data['last_time']


class TrainingManager:
    @dataclass
    class Info:
        experiences_per_step: int

        training_samples_cnt: int
        validation_sample_cnt: int

        max_epoch: int

    def __init__(self,
                 state_manager: StateManager,
                 sim: TradingEnvironment,
                 agent: ExplorationArbiter,
                 trainer: DQNTrainer,
                 provider: SampleProvider,
                 cfg: DQNTrainer.Config):
        self._sim = sim
        self._agent = agent
        self._trainer = trainer
        self._experience_provider = ExperienceProvider(state_manager,
                                                       provider,
                                                       agent,
                                                       sim)

        self._cfg = cfg

        self._training_thread = None
        self._abort = False

        self._watch = benchmark.Watch()

    def stop_training(self, blocking: bool):
        if self._training_thread is None:
            return

        self._abort = True

        if blocking:
            self._training_thread.join()

    def start_training_async(self, reporter: TrainingReporter):
        if self._training_thread is not None:
            raise RuntimeError('Training already in progress.')

        self._reporter = reporter
        self._reporter.start_update_thread()
        self._training_thread = threading.Thread(target=self._run_training)
        self._training_thread.start()

    def is_running(self):
        return (self._training_thread is not None
                and self._training_thread.is_alive())

    def get_info(self) -> Info:
        return TrainingManager.Info(
            self._cfg.iterations.experience_cnt,
            self._experience_provider.get_experience_cnt(ChunkType.TRAINING),
            self._experience_provider.get_experience_cnt(ChunkType.VALIDATION),
            self._cfg.iterations.max_epoch_cnt
        )

    def _run_training(self):
        epoch = 0
        self._abort = False

        self._watch.reset()

        while not self._abort and epoch < self._cfg.iterations.max_epoch_cnt:
            self._agent.update(epoch)
            self._train_and_validate_epoch()
            epoch += 1
        self._reporter.stop(True)
        self._training_thread = None

    def _train_and_validate_epoch(self):
        self._watch.start()
        self._experience_provider.start(ChunkType.TRAINING)

        while self._experience_provider.has_next():
            self._training_step()

        self._perform_validation()

        epoch_time, av = self._watch.stop('epoch', True, True, True)
        if self._reporter is not None:
            self._reporter.add_epoch_update(av, epoch_time)

    def _training_step(self):
        self._watch.stop('training_step_start')
        cnt = self._trainer.perform_exploration(
            self._cfg.iterations.experience_cnt,
            self._experience_provider.provide_experience)

        _ = self._trainer.perform_training(self._cfg.iterations.batch_size,
                                           self._cfg.iterations.batch_cnt,
                                           cuda=True)

        self._watch.stop('training_step')
        if self._reporter is not None:
            self._reporter.add_training_update(
                cnt,
                self._watch.get_av_time('training_step'))

    def _perform_validation(self):
        self._watch.stop('validation_start')
        self._experience_provider.start(ChunkType.VALIDATION)

        experiences = []

        while self._experience_provider.has_next():
            exp = self._experience_provider.provide_experience()

            if self._reporter is not None:
                experiences.append(exp)

            self._watch.stop('validation')
            if len(experiences) == 100:
                self._reporter.add_validation_update(
                    experiences,
                    self._watch.get_av_time('validation'))
                experiences = []
