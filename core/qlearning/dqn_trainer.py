import copy
from dataclasses import dataclass
from typing import Tuple, Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from core.qlearning.replay_buffer import ReplayBuffer, Experience
from utils.config_utils import assert_fields_in_dict


def perform_training_step(nn: torch.nn.Module,
                          optimizer: Optimizer,
                          loss: torch.nn.Module,
                          tr_data: Tuple[object, object]):
    input_data, target_data = tr_data

    optimizer.zero_grad()
    result_tensor = nn(input_data)
    loss_tensor = loss(result_tensor, target_data)
    loss_tensor.backward()
    optimizer.step()


def calculate_target_q(target_dqn: torch.nn.Module,
                       rewards: Tensor,
                       next_states: Tensor,
                       discount_factor: float) -> Tensor:
    """
    Calculates the Q_action target for given experiences.
    :param target_dqn: Target DQN for Q-Value calculation.
    :param rewards: [N] Rewards for choosing 'next_states'
    :param next_states: [N] follow states (DQNWrapper input)
    :param discount_factor: reduces impact from future states
    :return: Tensor containing one q-value per experience tuple
    """
    next_q_values = target_dqn(next_states)
    max_q_next, _ = torch.max(next_q_values, dim=1)
    return rewards + discount_factor * max_q_next


class DQNTrainer:
    """
        This class handles the training process for a Deep-Q-Network.

        The training is performed in episodes consisting of alternating
        exploration and training steps using a separated target dqn for
        determining the target Q-values. The target network parameters are
        updated after a minimum number of training samples.
    """

    @dataclass
    class ExperienceTuple:
        experience: Experience
        weight: float
        meta: object

    @dataclass
    class Config:
        @dataclass
        class Optimizer:
            type: str
            learning_rate: float
            weight_decay: float

        @dataclass
        class Iterations:
            max_epoch_cnt: int
            batch_cnt: int
            batch_size: int
            experience_cnt: int

        @dataclass
        class QLearning:
            discount_factor: float
            update_target_network_after: int
            replay_buffer_size: int

        optimizer: Optimizer
        iterations: Iterations
        qlearning: QLearning

        @staticmethod
        def from_dict(config_dict: dict):
            assert_fields_in_dict(config_dict,
                                  ['optimizer', 'iterations', 'qlearning',
                                   'exploration'])
            assert_fields_in_dict(config_dict['optimizer'],
                                  ['type', 'learning_rate', 'weight_decay'])
            assert_fields_in_dict(config_dict['iterations'],
                                  ['max_epoch_cnt', 'batch_cnt', 'batch_size',
                                   'experience_cnt'])
            assert_fields_in_dict(
                config_dict['qlearning'],
                ['discount_factor', 'update_target_network_after',
                 'replay_buffer_size'])

            return DQNTrainer.Config(
                DQNTrainer.Config.Optimizer(
                    config_dict['optimizer']['type'],
                    config_dict['optimizer']['learning_rate'],
                    config_dict['optimizer']['weight_decay']
                ),
                DQNTrainer.Config.Iterations(
                    config_dict['iterations']['max_epoch_cnt'],
                    config_dict['iterations']['batch_cnt'],
                    config_dict['iterations']['batch_size'],
                    config_dict['iterations']['experience_cnt']
                ),
                DQNTrainer.Config.QLearning(
                    config_dict['qlearning']['discount_factor'],
                    config_dict['qlearning']['update_target_network_after'],
                    config_dict['qlearning']['replay_buffer_size']
                ))

    @staticmethod
    def from_config(dqn: torch.nn.Module, config: Config) -> 'DQNTrainer':
        if config.optimizer.type != 'adam':
            raise ValueError('Only Adam optimizer is supported.')

        optimizer = torch.optim.Adam(dqn.parameters(),
                                     lr=config.optimizer.learning_rate,
                                     weight_decay=config.optimizer.weight_decay)
        replay_buffer = ReplayBuffer(config.qlearning.replay_buffer_size, True)
        return DQNTrainer(dqn, replay_buffer, optimizer,
                          config.qlearning.update_target_network_after,
                          config.qlearning.discount_factor)

    def __init__(self,
                 dqn: torch.nn.Module,
                 replay_buffer: ReplayBuffer,
                 optimizer: torch.optim.Optimizer,
                 update_target_network_after: int,
                 discount_factor: float):
        self._dqn = dqn
        self._target_dqn = copy.deepcopy(dqn)
        self._target_update_after = update_target_network_after
        self._cnt_till_target_update = update_target_network_after

        self._replay_buffer = replay_buffer

        self._optimizer = optimizer
        self._loss = nn.MSELoss()

        self._discount_factor = discount_factor

    def perform_exploration(
            self,
            cnt: int,
            experience_provider: Callable[[], ExperienceTuple]) \
            -> int:

        """
        Collects experience tuples into the replay buffer by iteratively
        sampling from experience_provider. Stops after explorations_per_episode
        experiences are sampled or sampler returns None.
        :param cnt: number of experiences to be sampled
        :param experience_provider: produces (Experience, weight) tuple.
        """
        experience_cnt = 0
        with torch.no_grad():
            self._dqn.train(False)
            for i in range(cnt):
                exp = experience_provider()
                if exp is not None:
                    self._replay_buffer.add_experience(exp.experience,
                                                       exp.weight)
                    experience_cnt += 1
                else:
                    break
        return experience_cnt

    def perform_training(self,
                         batch_size, batch_cnt, cuda) -> Tensor:
        """
        Performs training over training_batch_cnt mini-batches sampled from
        replay_buffer. Training consist of optimizer steps towards Q value
        produced by target net and updating target net after
        cnt_till_target_update training samples.
        """

        experiences = [self._replay_buffer.sample_experiences(
            batch_size,
            replace=True
        ) for _ in range(batch_cnt)]

        prev_states = [{k: Tensor(e['prev_state'][k]) for k in e['prev_state']}
                       for e in experiences]
        rewards = Tensor(np.concatenate([e['reward'] for e in experiences]))
        actions = [e['action'] for e in experiences]
        next_states = [{k: Tensor(e['next_state'][k]) for k in e['next_state']}
                       for e in experiences]
        if cuda:
            rewards = rewards.cuda()
            for e in prev_states:
                for k in e:
                    e[k] = e[k].cuda()
            for e in next_states:
                for k in e:
                    e[k] = e[k].cuda()
        loss_sum = 0

        self._dqn.train(True)
        self._dqn.requires_grad_(True)
        torch.set_grad_enabled(True)

        for batch_ix in range(batch_cnt):
            self._optimizer.zero_grad()
            # [Nx3]
            q = self._dqn(prev_states[batch_ix])
            # select chosen action q-value
            q_action = q[torch.arange(batch_size), actions[batch_ix]]
            # [Nx1]
            target_q = calculate_target_q(self._target_dqn,
                                          rewards[batch_ix],
                                          next_states[batch_ix],
                                          self._discount_factor)
            loss = self._loss(q_action, target_q)
            loss_sum += loss
            loss.backward()
            self._optimizer.step()

        self._cnt_till_target_update -= 1
        if self._cnt_till_target_update <= 0:
            self._cnt_till_target_update = self._target_update_after
            self._target_dqn.load_state_dict(self._dqn.state_dict())

        return loss / batch_cnt

    def get_replay_buffer(self) -> ReplayBuffer:
        return copy.deepcopy(self._replay_buffer)

    def get_target_dqn(self) -> nn.Module:
        return copy.deepcopy(self._target_dqn)
