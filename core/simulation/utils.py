from core.data.data_provider import ChunkType
from core.qlearning.q_arbiter import ExplorationArbiter
from core.simulation.data_classes import TransitionResult
import core.simulation.sample_provider as p
import core.simulation.simulation_environment as env


def perform_transition(agent: ExplorationArbiter,
                       simulation: env.SimulationEnvironment,
                       sample_provider: p.SampleProvider,
                       state: env.State,
                       mode: ChunkType,
                       cuda: bool) -> TransitionResult:

    with agent.explore(mode == ChunkType.TRAINING):
        action = agent.decide(state.to_tensor(cuda))

    new_context = simulation.perform_transition(
            state.samples,
            state.context.copy(),
            action,
            mode
        )

    new_samples = state.samples.copy()
    new_samples.update(sample_provider.get_context_samples(new_context))

    new_state = env.State(new_samples, new_context)

    reward = simulation.calculate_reward(state, new_state, action, mode)

    return TransitionResult(state, action, reward, new_state)
