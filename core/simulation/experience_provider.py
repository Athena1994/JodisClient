

from typing import Tuple
from core.data.data_provider import ChunkType
from core.qlearning.q_arbiter import Arbiter
from core.qlearning.replay_buffer import Experience
from core.simulation.sample_provider import SampleProvider
from core.simulation.simulation_environment import SimulationEnvironment, State
from core.simulation.state_manager import StateManager

class ExperienceProvider:
    def __init__(self, 
                 state_manager: StateManager,
                 sample_provider: SampleProvider,
                 agent: Arbiter,
                 simulation: SimulationEnvironment) -> None:

        self._state_manager = state_manager
        self._sample_provider = sample_provider
        self._agent = agent
        self._simulation = simulation

        self._running = False

        self._mode = None

    def _update_state(self, context: dict):
        self._state_manager.update_context(context)
        samples = self._state_manager.get_samples()
        self._sample_provider.update_values(samples)
        self._state_manager.update_samples(samples)

    def _advance_state(self, context: dict) -> State:
        self._state_manager.reset_state(context)
        self._state_manager.update_samples(
            self._sample_provider.get_next_samples()
        )
        return self._state_manager.get_state()

    def _start_next_episode(self):
        self._advance_state(
            self._simulation.on_episode_start(
                self._state_manager.get_context())
        )
        

    def start(self, type: ChunkType):
        if self._running:
            raise Exception("Experience provider already running")
        
        self._mode = type

        self._state_manager.reset_state(
            self._simulation.get_initial_context()
        )
        self._sample_provider.reset(type)
        self._start_next_episode()

        self._running = True

    def provide_experience(self) -> Tuple[Experience, float]:
        if not self._running:
            raise Exception("Experience provider not running")

        if self._sample_provider.current_episode_complete():
            try:
                self._start_next_episode()
            except StopIteration:
                self._running = False
                return None
            
        current_state = self._state_manager.get_state()

        action = self._agent.decide(current_state, 
                                    self._mode == ChunkType.TRAIN)
        next_state = self._advance_state(
            self._simulation.perform_transition(
                self._state_manager.get_samples(),
                self._state_manager.get_context(), 
                action, 
                self._mode
            )
        )
        reward = self._simulation.calculate_reward(current_state,
                                                   next_state,
                                                   action)

        self._update_state(
            self._simulation.on_action(self._state_manager.get_context(), 
                                       action, 
                                       self._mode)
        )

        exp = Experience(
            old_state=current_state,
            action=action, 
            reward=reward, 
            new_state=next_state)

        return exp, 1
