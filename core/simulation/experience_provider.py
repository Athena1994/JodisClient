import logging
from core.data.data_provider import ChunkType
from core.qlearning.dqn_trainer import DQNTrainer
from core.qlearning.q_arbiter import ExplorationArbiter
from core.simulation.sample_provider import SampleProvider
from core.simulation.simulation_environment import SimulationEnvironment
from core.simulation.state_manager import StateManager
from core.simulation.utils import perform_transition


class ExperienceProvider:
    def __init__(self,
                 state_manager: StateManager,
                 sample_provider: SampleProvider,
                 agent: ExplorationArbiter,
                 simulation: SimulationEnvironment) -> None:

        self._state_manager = state_manager
        self._sample_provider = sample_provider
        self._agent = agent
        self._simulation = simulation

        self._running = False

        self._mode = None

    def get_experience_cnt(self, type: ChunkType) -> int:
        return self._sample_provider.get_sample_cnt(type)

    def start(self, type: ChunkType):
        if self._running:
            logging.warn("restart running experience provider")

        self._mode = type

        self._state_manager.reset_state(
            self._simulation.get_initial_context()
        )
        self._sample_provider.reset(type)
        self._start_next_episode()

        self._running = True

    def provide_experience(self) -> DQNTrainer.ExperienceTuple:

        if not self._running:
            raise Exception("Experience provider not running")

        if self._sample_provider.current_episode_complete():
            try:
                self._start_next_episode()
            except StopIteration:
                self._running = False
                return None

        current_state = self._state_manager.get_state()

        result = perform_transition(
            self._agent,
            self._simulation,
            self._sample_provider,
            current_state,
            self._mode,
            True)

        self._advance_state(
            self._simulation.on_transition(result, self._mode))

        return DQNTrainer.ExperienceTuple(result.as_experience(), 1, result)

    def has_next(self) -> bool:
        return self._running

    def _advance_state(self, context: dict):
        self._state_manager.reset_state(context)

        samples = self._sample_provider.advance()
        context = self._simulation.on_new_samples(samples, context, self._mode)
        samples.update(self._sample_provider.get_context_samples(context))

        self._state_manager.update_samples(samples)
        self._state_manager.update_context(context)

    def _start_next_episode(self):
        self._advance_state(
            self._simulation.on_episode_start(
                self._state_manager.get_context(),
                self._mode)
        )
