

from typing import Tuple
from core.data.data_provider import DataProvider
from core.qlearning.q_arbiter import QSigArbiter
from core.qlearning.replay_buffer import Experience
from core.simulation.command_processor import CommandProcessor
from core.simulation.trading_simulation import TradingSimulation

"""
Context generated for each step of the simulation.

- "action": str 
    -> action to be evaluated

- "valid_action": bool 
    -> if the action is valid

- "previous_money": float 
    -> value of the wallet previous to the action

- "current_money": float 
    -> current value of the wallet
"""

class Action:
    BUY = 0
    SELL = 1
    HOLD = 2

class ExperienceProvider(TradingSimulation):
    def __init__(self, 
                 data_provider: DataProvider,
                 agent: QSigArbiter,
                 sim_config: dict) -> None:
        super().__init__(data_provider, sim_config)

        self._agent = agent
        self._current_state = None
        self._reward_processor = CommandProcessor(sim_config['reward'])

    def start_session(self, type: str) -> None:
        return super().start_session(type)

    def provide_experience(self) -> Tuple[Experience, float]:
        if (self._current_state is None
        or  self._episode != self._current_state.episode):
            self._current_state = self.get_next_state()

        money_pre_action = self._current_state.simulation['money']

        action = self._agent.decide(self._current_state, explore=self.is_training) 

        if action == Action.BUY:
            valid_action = self.buy(self._current_state.context) is not None  
            action_str = 'buy'
        elif action == Action.SELL:
            valid_action = self.sell(self._current_state.context) is not None  
            action_str = 'sell'
        else:
            valid_action = True
            action_str = 'hold'

        new_state = self.get_next_state()
        if new_state is None:
            return None

        money_post_action = new_state.context['money']
        reward = self._reward_processor.execute({
            'action': action_str,
            'valid_action': valid_action,
            'previous_money': money_pre_action,
            'current_money': money_post_action
        })


        exp = Experience(self._current_state, action, reward, new_state)

        self._current_state = new_state

        return exp, 1



