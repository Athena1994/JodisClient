

import copy
import math
from typing import Tuple
import unittest

import numpy as np
from torch import Tensor, nn
import torch

from core.nn.dynamic_nn import DynamicNN
from core.qlearning.q_arbiter import DQNWrapper, DeepQFunction, EpsilonGreedyArbiter
from core.qlearning.replay_buffer import Experience, ReplayBuffer
from core.qlearning.dqn_trainer \
    import DQNTrainer, calculate_target_q


class DummyReplayBuffer(ReplayBuffer):
    def __init__(self, size):
        super().__init__(size, False)

    def sample_experiences(self, size: int | Tuple, replace: bool) -> dir:

        cnt = size[-1] if isinstance(size, Tuple) else size

        if self._cnt == 0:
            raise Exception("can not sample on empty replay buffer")
        if self._cnt < cnt and not replace:
            raise Exception("not enough samples in replace buffer")

        choices = np.arange(size[1])
        res = {
            'prev_state': np.repeat(self._prev_state_buffer[None, choices],
                                    size[0], 0),
            'action': np.repeat(self._action_buffer[None, choices],
                                size[0], 0),
            'reward': np.repeat(self._reward_buffer[None, choices],
                                size[0], 0),
            'next_state': np.repeat(self._next_state_buffer[None, choices],
                                    size[0], 0)
        }
        return res


class TestTrainer(unittest.TestCase):

    def test_init(self):
        cfg_dict = {
            "qlearning": {
                "replay_buffer_size": 8192,
                "update_target_network_after": 4096,
                "discount_factor": 0.99
            },
            "exploration": {
                "sigma": 0.01
            },
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0001
            },
            "iterations": {
                "max_epoch_cnt": 128,
                "batch_cnt": 64,
                "batch_size": 32,
                "experience_cnt": 2048
            }
        }
        cfg = DQNTrainer.Config.from_dict(cfg_dict)

        dummy_nn = nn.Linear(8, 32)

        trainer = DQNTrainer.from_config(dummy_nn, cfg)
        self.assertIsNotNone(trainer)

    def test_dict_states(self):

        def provide_experience():
            return DQNTrainer.ExperienceTuple(Experience(
                {
                    'a': Tensor(np.random.random((input_cfg.input_window, 2))),
                    'b': Tensor(np.random.random((input_cfg.input_window, 3)))
                },
                0, 0,
                {
                    'a': Tensor(np.random.random((input_cfg.input_window, 2))),
                    'b': Tensor(np.random.random((input_cfg.input_window, 3)))
                }), 1, None)

        input_cfg = DynamicNN.Config.Input.from_dict({
            "input_window": 5,
            "data": [{
                "key": "a",
                "type": "asset",
                "params": {
                    "asset": {
                        "name": "foo",
                        "source": "bar",
                        "interval": "foobar"
                    },
                    "include": ["v1", "v2"],
                    "indicators": [],
                    "normalizer": {}
                }
            }, {
                "key": "b",
                "type": "asset",
                "params": {
                    "asset": {
                        "name": "foo",
                        "source": "bar",
                        "interval": "foobar"
                    },
                    "include": ["v1", "v2", "v3"],
                    "indicators": [],
                    "normalizer": {}
                }
            }]
        })

        nn_cfg = DynamicNN.Config.from_dict({
            'units': [{
                'name': 'in',
                "input": 'a',
                'type': 'LSTM',
                'params': {
                    'num_layers': 1,
                    'hidden_size': 5
                }
            }, {
                'name': 'in2',
                "input": 'b',
                'type': 'Sequence',
                'params': [
                    {'type': 'Linear', 'size': 10}
                ]
            }, {
                'name': 'out',
                'input': 'in',
                'type': 'Sequence',
                'params': [
                    {'type': 'Linear', 'size': 3}
                ]
            }
            ],
            'output': 'out'
        })
        nn = DynamicNN(nn_cfg, input_cfg).cuda()

        trainer_cfg = DQNTrainer.Config(
            DQNTrainer.Config.Optimizer('adam', 0.001, 0.0001),
            DQNTrainer.Config.Iterations(
                max_epoch_cnt=5,
                batch_cnt=2,
                batch_size=3,
                experience_cnt=4),
            DQNTrainer.Config.QLearning(0.99, 10, 7),
            DQNTrainer.Config.Exploration(0.1)
        )

        trainer = DQNTrainer.from_config(nn, trainer_cfg)

        trainer.perform_exploration(cnt=trainer_cfg.iterations.experience_cnt,
                                    experience_provider=provide_experience)

        exp_batch = trainer.get_replay_buffer().sample_experiences(16, True)
        self.assertTrue('a' in exp_batch['prev_state']
                        and 'b' in exp_batch['prev_state'])
        self.assertEqual(exp_batch['prev_state']['a'].shape, (16, 5, 2))
        self.assertEqual(exp_batch['prev_state']['b'].shape, (16, 5, 3))

        trainer.perform_training(batch_size=trainer_cfg.iterations.batch_size,
                                 batch_cnt=trainer_cfg.iterations.batch_cnt,
                                 cuda=True)


    # def test_training_step(self):
    #     print(os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    #     input_tensor = Tensor([(0, 0), (0, 1), (1, 0), (1, 1)])
    #     target_tensor = Tensor([(1, 0), (0, 1), (0, 1), (1, 0)])

    #     nn = torch.nn.Sequential(
    #         torch.nn.Linear(2, 10),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(10, 2),
    #         torch.nn.Sigmoid(),
    #     )
    #     optimizer = torch.optim.Adam(params=nn.parameters(), lr=0.1)
    #     loss = torch.nn.MSELoss()

    #     for _ in range(100):
    #         perform_training_step(nn,
    #                               optimizer,
    #                               loss,
    #                               (input_tensor, target_tensor))

    #     result = torch.round(nn(input_tensor))
    #     self.assertListEqual([list(v) for v in target_tensor],
    #                          [list(v) for v in result])

    #     # repeat with cuda
    #     nn = nn.cuda()
    #     optimizer = torch.optim.Adam(params=nn.parameters(), lr=0.1)
    #     loss = loss.cuda()

    #     for _ in range(1):
    #         perform_training_step(nn,
    #                               optimizer,
    #                               loss,
    #                               (input_tensor.cuda(), target_tensor.cuda()))

    #     result = torch.round(nn(input_tensor.cuda()))
    #     self.assertListEqual([list(v) for v in target_tensor],
    #                          [list(v) for v in result])

    def test_calculate_target_q(self):
        class DummyTargetQNN(torch.nn.Module):
            def __init__(self, data):
                super().__init__()
                self._data = Tensor(data)

            def forward(self, x):
                res = self._data[x.long()]
                return torch.flatten(res, 1, 2)

        value_cnt = 256
        q_val_cnt = 4
        q_values_per_state = np.random.uniform(size=(value_cnt, q_val_cnt))
        max_q_per_state = np.max(q_values_per_state, axis=1)
        nn = DummyTargetQNN(q_values_per_state)

        sample_cnt = 512
        # input_states = np.random.randint(0, value_cnt, (sample_cnt, 1))
        rewards = np.random.uniform(-100, 100, sample_cnt)
        next_states = np.random.randint(0, value_cnt, (sample_cnt, 1))

        discount_factor = 0.5
        expected_q = rewards \
            + discount_factor * max_q_per_state[next_states.flatten()]
        target_q = calculate_target_q(nn,
                                      Tensor(rewards),
                                      Tensor(next_states),
                                      discount_factor)
        self.assertTrue(np.allclose(list(expected_q), list(target_q)))

    def test_dqn_perform_exploration_step(self):
        replay_buffer_size = 256
        explore_episode_len = 64

        replay_buffer = ReplayBuffer(replay_buffer_size, False)
        dqn = nn.Linear(8, 32)
        trainer = DQNTrainer(dqn,
                             replay_buffer,
                             optimizer=None,
                             update_target_network_after=0,
                             discount_factor=0.9)

        test_replay_buffer = ReplayBuffer(replay_buffer_size, False)

        # returns random sample and asserts trainer replay buffer
        def experience_provider():
            test_exp = [e.old_state
                        for e in test_replay_buffer.get_experiences()]
            trainer_exp \
                = [e.old_state
                   for e in trainer.get_replay_buffer().get_experiences()]
            for r, c in zip(trainer_exp, test_exp):
                self.assertListEqual(list(c[0]), list(r[0]))
                self.assertListEqual(list(c[1]), list(r[1]))
            experience = Experience((np.random.random(8), np.random.random(32)),
                                    0, 0, None)
            test_replay_buffer.add_experience(experience, 1)
            return DQNTrainer.ExperienceTuple(experience, 1, None)
        trainer.perform_exploration(explore_episode_len,
                                    experience_provider)

        # assert target and live dqn are independent of each other
        for p1, p2 in zip(trainer.get_target_dqn().parameters(),
                          dqn.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        optimizer = torch.optim.Adam(dqn.parameters())
        y = dqn.forward(Tensor(np.random.random((100, 8))))
        lv = nn.MSELoss()(y, Tensor(np.random.random((100, 32))))
        lv.backward()
        optimizer.step()
        for p1, p2 in zip(trainer.get_target_dqn().parameters(),
                          dqn.parameters()):
            self.assertFalse(torch.equal(p1, p2))

    def test_perform_training_session(self):
        """
            Train nn with trainer for one epoch and compare parameters with
            manually trained network. Also ensure target network is updated
            correctly.
        """
        #
        # test configuration and data
        #
        input_cnt = 2
        action_cnt = 3
        batch_size = 64
        discount = 0.99

        torch.manual_seed(0)
        np.random.seed(0)
        torch.use_deterministic_algorithms(True)

        test_experiences = {
            'state': np.random.random((batch_size, input_cnt)),
            'action': np.random.randint(0, action_cnt, batch_size),
            'reward': np.random.random(batch_size),
            'next': np.random.random((batch_size, input_cnt)),
        }

        #
        # prepare
        #
        base_nn = nn.Linear(input_cnt, action_cnt)
        compare_nn = nn.Linear(input_cnt, action_cnt)
        compare_nn.load_state_dict(base_nn.state_dict())

        original_weights = copy.deepcopy(base_nn.weight)

        optimizer = torch.optim.Adam(base_nn.parameters(), 0.9)
        compare_optimizer = torch.optim.Adam(compare_nn.parameters(), 0.9)

        def convert(state):
            return torch.stack(tuple(state)),
        dqn = DQNWrapper(base_nn, convert)
        replay_buffer = DummyReplayBuffer(batch_size)
        trainer = DQNTrainer(dqn,
                             replay_buffer,
                             optimizer,
                             update_target_network_after=batch_size + 1,
                             discount_factor=discount)
        #
        # fill replay buffer
        #
        ix = 0  # index for experience provider

        def experience_provider():
            nonlocal ix
            self.assertLessEqual(ix, batch_size)
            exp = Experience(
                Tensor(test_experiences['state'][ix]),
                test_experiences['action'][ix],
                test_experiences['reward'][ix],
                Tensor(test_experiences['next'][ix]))
            ix += 1
            return exp, 1
        trainer.perform_exploration(batch_size, experience_provider)

        # assert comparison training data and data provided to trainer are same
        rb_data = replay_buffer.sample_experiences((1, 64), True)
        rb_data = torch.stack(tuple(rb_data['prev_state'][0]))
        self.assertEqual(len(trainer.get_replay_buffer()), batch_size)
        training_data = Tensor(test_experiences['state'])
        self.assertTrue(torch.equal(rb_data, training_data))

        #
        # manually calculate updated network weights
        #
        target_nn = nn.Linear(input_cnt, action_cnt)
        target_nn.load_state_dict(compare_nn.state_dict())
        comp_target_q = calculate_target_q(target_nn,
                                           Tensor(test_experiences['reward']),
                                           Tensor(test_experiences['next']),
                                           discount)
        self.assertEqual(tuple(comp_target_q.shape), (batch_size, ))

        compare_optimizer.zero_grad()
        # [batch_size x action_cnt]
        q = compare_nn.forward(training_data)
        # [batch_size]
        q_action = q[torch.arange(batch_size), test_experiences['action']]
        loss = torch.nn.MSELoss()
        compare_loss = loss(q_action, comp_target_q)
        compare_loss.backward()
        # assert weights are identical pre-training
        self.assertTrue(torch.equal(base_nn.weight,
                                    compare_nn.weight))
        compare_optimizer.step()
        # assert weights changed
        self.assertFalse(torch.equal(base_nn.weight,
                                     compare_nn.weight))

        #
        # use trainer to update weights
        #
        trainer_loss = trainer.perform_training(batch_size, 1, False)
        self.assertEqual(compare_loss, trainer_loss)
        self.assertTrue(torch.equal(base_nn.weight,
                                    compare_nn.weight))

        #
        # assert second training session updates target network
        #
        self.assertTrue(torch.equal(trainer.get_target_dqn()._nn.weight,
                                    original_weights))
        trainer_loss = trainer.perform_training(batch_size, 2, False)
        self.assertFalse(torch.equal(trainer.get_target_dqn()._nn.weight,
                                     original_weights))
        self.assertTrue(torch.equal(trainer.get_target_dqn()._nn.weight,
                                    dqn._nn.weight))

    def test_dqn(self):
        #   -----------------------------
        #   | 50| x | x | x | x | W | E |
        #   | W | W | W | W | x | W | x |
        #   | x | x | x | W | x | W | x |
        #   | x | W | x | W | x | W | x |
        #   | x | W | x | x | x | W | x |
        #   | 5 | W | 5 | W | x | x | x |
        #   -----------------------------

        np.random.seed(42)
        torch.random.manual_seed(42)

        learning_rate = 0.001
        discount = 0.95

        reward_map = np.array([[100,  -1,  -1,  -1, -1, -50, 42],
                               [-50, -50, -50, -50, -1, -50,  -1],
                               [ -1,  -1,  -1, -50, -1, -50,  -1], # noqa
                               [ -1, -50,  -1, -50, -1, -50,  -1], # noqa
                               [ -1, -50,  -1,  -1, -1, -50,  -1], # noqa
                               [100, -50, 100, -50, -1,  -1,  -1]])
        reward_map = reward_map.transpose(1, 0)

        print(reward_map)

        def make_map_tensor(r_map, x, y):
            wall_tensor = np.zeros(r_map.shape)
            wall_tensor[r_map == -50] = 1
            wall_tensor = Tensor(wall_tensor[:, :])
            exit_tensor = np.zeros(r_map.shape)
            exit_tensor[r_map == 42] = 1
            exit_tensor = Tensor(exit_tensor[:, :])
            treasure_tensor = np.zeros(r_map.shape)
            treasure_tensor[r_map == 100] = 1
            treasure_tensor = Tensor(treasure_tensor[:, :])
            location_tensor = np.zeros(r_map.shape)
            location_tensor[x, y] = 1
            location_tensor = Tensor(location_tensor[:, :])

            return torch.stack((wall_tensor, exit_tensor,
                                treasure_tensor, location_tensor))

        def state_to_input_tensor(state):
            if isinstance(state, Tensor):
                return state.reshape(1, *state.shape).cuda(),
            else:
                res = torch.stack(tuple(state)).cuda()
                return res,

        def observe_and_move(rmap, loc, action_arbiter, explore=True):
            state = make_map_tensor(rmap, *loc)
            action = action_arbiter.decide(state, explore=explore)[0]

            # move location
            d_loc = [(-1, 0), (0, -1), (1, 0), (0, 1)]
            new_loc = (loc[0] + d_loc[action][0],
                       loc[1] + d_loc[action][1])
            # trying to leave boundaries
            if (new_loc[0] < 0 or new_loc[0] >= width
                    or new_loc[1] < 0 or new_loc[1] >= height):
                reward = -100
                new_loc = loc
            else:
                map_val = rmap[new_loc[0], new_loc[1]]
                if map_val == 42:
                    reward = 50
                elif map_val > 0:
                    reward = 20
                    rmap[new_loc[0], new_loc[1]] = -1
                elif map_val == -1:
                    reward = -1
                else:
                    reward = -200

            return new_loc, action, reward

        current_loc = None
        current_map = None
        loc_lst = None

        def explore_state():
            nonlocal current_loc, current_map, loc_lst
            if current_loc is None:
                current_loc = np.stack((np.random.randint(0, width),
                                        np.random.randint(0, height)))
                current_map = copy.deepcopy(reward_map)
                loc_lst = [current_loc]

            new_loc, action, reward = observe_and_move(current_map,
                                                       current_loc,
                                                       arbiter)
            exp = Experience(
                make_map_tensor(current_map, current_loc[0], current_loc[1]),
                action,
                reward,
                make_map_tensor(current_map, new_loc[0], new_loc[1]))

            result = exp, abs(math.atan(reward))+1
            loc_lst.append(new_loc)

            def eq(l1, l2):
                return l1[0] == l2[0] and l1[1] == l2[1]

            if len(loc_lst) >= 4:
                if (eq(loc_lst[-1], loc_lst[-3]) and
                        eq(loc_lst[-2], loc_lst[-4])):
                    current_loc = None
                    return result

            if current_loc[0] == new_loc[0] and current_loc[1] == new_loc[1]:
                current_loc = None
            elif current_map[new_loc[0], new_loc[1]] == 42:
                current_loc = None
            else:
                current_loc = new_loc

            return result

        explorations_per_episode = 512
        batch_size = 128
        mini_batch_cnt = 4
        target_update_after = 3000

        width = reward_map.shape[0]
        height = reward_map.shape[1]

        dqn = DQNWrapper(torch.nn.Sequential(
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 128, 3, 1, 1),
            torch.nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 32, 3, 1, 1),
            torch.nn.ReLU(),
            nn.Flatten(),
            torch.nn.Dropout(),
            torch.nn.Linear(32 * width*height, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4),
        ), state_to_input_tensor).cuda()
        for p in dqn.parameters():
            if isinstance(p, nn.Linear):
                torch.nn.init.xavier_uniform(p.weight)
                torch.nn.init.xavier_uniform(p.bias)

        arbiter = EpsilonGreedyArbiter(DeepQFunction(dqn), 0.3)
        replay_buffer = ReplayBuffer(4096, False)
        optimizer = torch.optim.Adam(dqn.parameters(), lr=learning_rate,
                                     weight_decay=0.0001)

        trainer = DQNTrainer(dqn,
                             replay_buffer,
                             optimizer,
                             update_target_network_after=target_update_after,
                             discount_factor=discount)

        loss_lst = []
        min_loss = 99999999
        episode_cnt = 5
        for episode in range(episode_cnt):
            trainer.perform_exploration(explorations_per_episode,
                                        explore_state)
            loss = trainer.perform_training(batch_size, mini_batch_cnt, True)
            loss_lst.append(float(loss))
            if len(loss_lst) > 10:
                loss_lst = loss_lst[1:]
                min_loss = min((min_loss, float(loss)))
            print("min:", min_loss,
                  "av:", sum(loss_lst)/len(loss_lst),
                  "current:", float(loss))

            ix = 0
            test_map = copy.deepcopy(reward_map)
            test_loc = (2, 2)
            reward_sum = 0
            lst = []
            loc_lst = [test_loc]
            while ix < 100:
                old_loc = test_loc
                test_loc, action, reward \
                    = observe_and_move(test_map, old_loc, arbiter, False)
                lst += [(old_loc, action, reward, test_loc)]
                loc_lst.append(test_loc)
                reward_sum += reward

                def eq(l1, l2):
                    return l1[0] == l2[0] and l1[1] == l2[1]

                if len(loc_lst) >= 4:
                    if (eq(loc_lst[-1], loc_lst[-3]) and
                            eq(loc_lst[-2], loc_lst[-4])):
                        break

                if test_map[test_loc[0], test_loc[1]] == 42\
                        or test_loc == old_loc:
                    break
                ix += 1

            dir = ["L", "T", "R", "B"]

            t = [f"--{dir[e[1]]}({e[2]})-->{e[3]}" for e in lst]
            print(reward_sum, ''.join([str(lst[0][0])] + t))
