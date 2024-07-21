
from enum import Enum
from typing import Dict

import numpy as np

from core.data.data_provider import ChunkType, Sample
from core.simulation.simulation_environment import SimulationEnvironment, State
from program.exchange_manager import ExchangeDirection, Exchanger


class ActionType(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2

    @staticmethod
    def from_int(value: int):
        if value == 0:
            return ActionType.BUY
        elif value == 1:
            return ActionType.SELL
        elif value == 2:
            return ActionType.HOLD
        else:
            raise Exception(f"Invalid action type {value}")


class TradingEnvironment(SimulationEnvironment):

    def _get_constant(self, name: str) -> any:
        if name not in self._constants:
            raise Exception(f"Missing constant {name}")
        return self._constants[name]

    def __init__(self, config: dict, exchanger: Exchanger) -> None:
        super().__init__()
        if 'constants' not in config:
            raise Exception("Missing constants in configuration")

        self._constants = config['constants']
        self._exchanger = exchanger

    def get_initial_context(self) -> dict:
        return {
            'asset': {'BTC': 0,
                      'EUR': self._get_constant('initial_balance')},
            'position_open': False,
            'buy_in_price': None
        }

    def on_episode_start(self, context: dict, mode: ChunkType) -> dict:
        if mode == ChunkType.TRAINING or mode == ChunkType.VALIDATION:
            return self.get_initial_context()
        else:  # if TEST mode: reset open positions but retain balance
            if context['position_open']:
                context['asset']['BTC'] = 0
                context['money']['EUR'] = context['buy_in_price']
                context['position_open'] = False
                context['buy_in_price'] = None
            return context

    def on_new_samples(self,
                       samples: Dict[str, Sample],
                       context: dict,
                       mode: ChunkType) -> dict:
        if not context['position_open']:
            return context

        old_balance = context['buy_in_price']

        details = self._exchanger.perform_exchange(
            'EUR', 'BTC', old_balance, ExchangeDirection.SELL)

        context['balance'] = details.currency_balance - old_balance

        return context

    def perform_transition(self,
                           samples: Dict[str, Sample],
                           context: dict,
                           action: int,
                           mode: ChunkType) -> dict:

        if context['position_open']:
            if action == ActionType.BUY:
                context['invalid_action'] = True
                return context
            elif action == ActionType.SELL:
                details = self._exchanger.perform_exchange(
                    'EUR', 'BTC',
                    context['asset']['BTC'], ExchangeDirection.SELL)
                context['asset']['BTC'] = 0
                context['asset']['EUR'] = details.currency_balance
                context['position_open'] = False
                context['buy_in_price'] = None
        else:
            if action == ActionType.BUY:
                details = self._exchanger.perform_exchange(
                    'EUR', 'BTC',
                    context['asset']['EUR'], ExchangeDirection.BUY)
                context['buy_in_price'] = context['asset']['EUR']
                context['asset']['BTC'] = details.asset_balance
                context['asset']['EUR'] = 0
                context['position_open'] = True
            elif action == ActionType.SELL:
                context['invalid_action'] = True

        return context

    def calculate_reward(self,
                         current_state: State,
                         next_state: State,
                         action: int,
                         mode: ChunkType) -> float:

        if 'invalid_action' in next_state.context:
            return self._get_constant('invalid_action_penalty')
        if action == ActionType.HOLD:
            if current_state.context['position_open']:
                return self._get_constant('hold_penalty')
            else:
                return self._get_constant('wait_penalty')
        if action == ActionType.BUY:
            return self._get_constant('buy_reward')
        if action == ActionType.SELL:
            return (np.log(current_state.context['balance'])
                    / np.log(1.01))

    def on_action(self,
                  context: dict,
                  action: int,
                  mode: ChunkType) -> dict:
        if "invalid_action" in context:
            del context["invalid_action"]

        if action == ActionType.SELL and mode == ChunkType.TRAINING:
            context['asset']['EUR'] = self._constants['initial_balance']
