
from dataclasses import dataclass
from enum import Enum
from typing import Dict

import numpy as np

from core.data.data_provider import ChunkType, Sample
from core.simulation.simulation_environment import SimulationEnvironment, State
from core.simulation.data_classes import TransitionResult
from program.exchange_manager import ExchangeDirection, Exchanger
from utils.config_utils import assert_fields_in_dict


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

    @dataclass
    class Config:
        constants: dict

        @staticmethod
        def from_dict(conf: dict) \
                -> 'TradingEnvironment.Config':
            assert_fields_in_dict(conf, ['constants'])
            return TradingEnvironment.Config(conf['constants'])

    def _get_constant(self, name: str) -> any:
        if name not in self._constants:
            raise Exception(f"Missing constant {name}")
        return self._constants[name]

    def __init__(self, cfg: Config, exchanger: Exchanger) -> None:
        super().__init__()

        required_constants = ['initial_balance',
                              'buy_reward',
                              'hold_penalty',
                              'wait_penalty',
                              'invalid_action_penalty']
        for constant in required_constants:
            if constant not in cfg.constants:
                raise Exception(f"Missing constant {constant} in configuration")

        self._constants = cfg.constants
        self._exchanger = exchanger

    def get_initial_context(self) -> dict:
        return {
            'asset': {'BTC': 0,
                      'EUR': self._get_constant('initial_balance')},
            'buy_in_price': None,
            'position_open': False,
            'balance': 1
        }

    def on_episode_start(self, context: dict, mode: ChunkType) -> dict:
        """
            if TRAINING or VALIDATION mode:
                reset everything to initial context
            if TEST mode: reset open positions but retain balance
        """

        if mode == ChunkType.TRAINING:
            return self.get_initial_context()
        else:  # if TEST mode: reset open positions but retain balance
            if context['position_open']:
                context['asset']['BTC'] = 0
                context['asset']['EUR'] = context['buy_in_price']
                context['position_open'] = False
                context['buy_in_price'] = None
                context['balance'] = 1
            return context

    def update_balance(self, context: dict, samples: Dict[str, Sample]):
        details = self._exchanger.prepare_exchange(
            'EUR', 'BTC',
            context['asset']['BTC'], ExchangeDirection.SELL,
            samples)

        context['balance'] = details.currency_balance / context['buy_in_price']

    def on_new_samples(self,
                       samples: Dict[str, Sample],
                       context: dict,
                       mode: ChunkType) -> dict:
        """
            iff position is open:
                balance = proceeds / buy_in_price
        """

        if not context['position_open']:
            return context

        self.update_balance(context, samples)

        return context

    def perform_transition(self,
                           samples: Dict[str, Sample],
                           context: dict,
                           action: int,
                           mode: ChunkType) -> dict:
        """
            on invalid action:
                set 'invalid_action' and return
            on BUY:
                - perform exchange with entire EUR balance
                - set buy_in_price to EUR balance
                - set position_open to True
                - update asset.EUR|BTC
            on SELL:
                - perform exchange with entire BTC balance
                - set position_open to False
                - set buy_in_price to None
                - update asset.EUR|BTC
                - reset balance to 1
        """

        if context['position_open']:
            if action == ActionType.BUY.value:
                context['invalid_action'] = True
                return context
            elif action == ActionType.SELL.value:
                details = self._exchanger.prepare_exchange(
                    'EUR', 'BTC',
                    context['asset']['BTC'], ExchangeDirection.SELL,
                    samples
                    )
                context['asset']['BTC'] = 0
                context['asset']['EUR'] = details.currency_balance
                context['position_open'] = False
                context['buy_in_price'] = None
                context['balance'] = 1
        else:
            if action == ActionType.BUY.value:
                details = self._exchanger.prepare_exchange(
                    'EUR', 'BTC',
                    context['asset']['EUR'], ExchangeDirection.BUY, samples)
                context['buy_in_price'] = context['asset']['EUR']
                context['asset']['BTC'] = details.asset_balance
                context['asset']['EUR'] = 0
                context['position_open'] = True
#                self.update_balance(context, samples)
            elif action == ActionType.SELL.value:
                context['invalid_action'] = True

        return context

    def calculate_reward(self,
                         current_state: State,
                         next_state: State,
                         action: int,
                         mode: ChunkType) -> float:
        """
            {invalid_action_penalty} iff invalid action
            {hold_penalty} iff hold on open position
            {wait_penalty} iff hold on closed position
            {buy_reward} iff buying
            log(balance) iff selling
        """

        if 'invalid_action' in next_state.context:
            return self._get_constant('invalid_action_penalty')
        if action == ActionType.HOLD.value:
            if current_state.context['position_open']:
                return self._get_constant('hold_penalty')
            else:
                return self._get_constant('wait_penalty')
        if action == ActionType.BUY.value:
            return self._get_constant('buy_reward')
        if action == ActionType.SELL.value:
            return (np.log(current_state.context['balance'])
                    / np.log(1.10))

        raise Exception(f"Invalid action {action}")

    def on_transition(self,
                      result: TransitionResult,
                      mode: ChunkType) -> dict:
        """
            - remove invalid action from context
            - reset balance if selling in training mode
        """

        context = result.new_state.context.copy()

        if "invalid_action" in context:
            del context["invalid_action"]

        if (result.action == ActionType.SELL.value
           and mode == ChunkType.TRAINING):
            context['asset']['EUR'] = self._constants['initial_balance']

        return context
