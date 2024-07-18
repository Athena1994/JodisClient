

import unittest

import pandas as pd
from core.simulation.sample_provider import State
from program.experience_provider import *

class DummyArbiter(Arbiter):    
    def __init__(self):
        self._action = None

    def set_next_action(self, action: int) -> None:
        self._action = action

    def decide(self, state: State, explore: bool) -> int:
        return self._action

class TestExperienceProvider(unittest.TestCase):
        
    def test_init(self):
        sim_config = {
        "general": {
            "initial_balance": 1000
        },

        "reward": {
            "process": [
                {"!": "if", "condition": "not {valid_action}", "then": [
                    {"!": "return", "value": "{invalid_action_penalty}"}
                ]},

                {"!": "switch", "condition": "{action}", "cases": [
                    {"value": "buy", "then": [
                        {"!": "store", "name": "position_open"},
                        {"!": "store", "name": "position", "value": "{previous_money}"},
                        {"!": "return", "value": "{buy_reward}"}
                    ]},
                    {"value": "sell", "then": [
                        {"!": "clear", "name": "position_open"},
                        {"!": "return", "value": "{current_money} - {position}"}
                    ]},
                    {"value": "hold", "then": [
                        {"!": "if", "condition": "'position_open' in env", "then":[
                            {"!": "return", "value": "{hold_penalty}"}
                        ], "else": [
                            {"!": "return", "value": "{wait_penalty}"}
                        ]}
                    ]}                    
                ]}
            ],

            "constants": {
                "invalid_action_penalty": -10,
                "buy_reward": 2,
                "hold_penalty": 0,
                "wait_penalty": -1
            }
        }}

        chunks = [i // 10 for i in range(100)]
        chunks[10] = 0
        chunks[11] = 0
        df = pd.DataFrame(
            {
                "date": pd.date_range(start='1/1/2021', periods=100),
                "close": range(1, 101),
                "chunk": chunks,
                "chunk_type": [0]*60 + [1]*20 + [2]*20, 
            }
        )
        arbiter = DummyArbiter()
        data_provider = AssetProvider(df, ['close'], 'none', 5)
        experience_provider = ExperienceProvider(data_provider, 
                                                 arbiter,
                                                 sim_config)
        
        for _ in range(2):
            experience_provider.start_session('tr')
        
            arbiter.set_next_action(Action.BUY)
            exp = experience_provider.provide_experience() #(0; 5->6/11)
            self.assertIsNotNone(exp)
            exp, weight = exp
            self.assertEqual(exp.old_state.simulation ['money'], 1000)
            self.assertEqual(exp.old_state.simulation['asset'], 0)
            self.assertEqual(exp.action, Action.BUY)
            self.assertEqual(exp.reward, 2)
            self.assertEqual(exp.new_state.simulation['money'], 0)
            self.assertEqual(exp.new_state.simulation['asset'], 1000 / 5)

            #6
            exp = experience_provider.provide_experience() #(0; 6->7/11)
            self.assertIsNotNone(exp)
            exp, weight = exp
            self.assertEqual(exp.old_state.simulation ['money'], 0)
            self.assertEqual(exp.old_state.simulation['asset'], 200)
            self.assertEqual(exp.action, Action.BUY)
            self.assertEqual(exp.reward, -10)
            self.assertEqual(exp.new_state.simulation['money'], 0)
            self.assertEqual(exp.new_state.simulation['asset'], 200)

            arbiter.set_next_action(Action.HOLD)
            exp = experience_provider.provide_experience() #(0; 7->8/11)
            self.assertIsNotNone(exp)
            exp, weight = exp
            self.assertEqual(exp.old_state.simulation ['money'], 0)
            self.assertEqual(exp.old_state.simulation['asset'], 200)
            self.assertEqual(exp.action, Action.HOLD)
            self.assertEqual(exp.reward, 0)
            self.assertEqual(exp.new_state.simulation['money'], 0)
            self.assertEqual(exp.new_state.simulation['asset'], 200)

            env = experience_provider._reward_processor._working_env

            self.assertTrue("position_open" in env)
            self.assertTrue("position" in env)
            self.assertEqual(env['position'], 1000)

            arbiter.set_next_action(Action.SELL)
            exp = experience_provider.provide_experience()  #(0; 8->9/11)
            self.assertIsNotNone(exp)
            exp, weight = exp
            self.assertEqual(exp.old_state.episode, exp.new_state.episode)
            self.assertEqual(exp.new_state.episode, 0)
            self.assertEqual(exp.old_state.simulation ['money'], 0)
            self.assertEqual(exp.old_state.simulation['asset'], 200)        
            self.assertEqual(exp.action, Action.SELL)
            self.assertEqual(exp.reward, 
                            (200 * exp.old_state.context['close']) - 1000)
            self.assertEqual(exp.new_state.simulation['money'], 1000)
            self.assertEqual(exp.new_state.simulation['asset'], 0)

            self.assertFalse("position_open" in experience_provider
                                            ._reward_processor
                                            ._working_env)

            arbiter.set_next_action(Action.HOLD)   
            exp = experience_provider.provide_experience()  #(0; 9->10/11)
            self.assertIsNotNone(exp)
            exp, weight = exp
            self.assertEqual(exp.old_state.episode, exp.new_state.episode)
            self.assertEqual(exp.new_state.episode, 0)
            self.assertEqual(exp.old_state.simulation ['money'], 1000)
            self.assertEqual(exp.old_state.simulation['asset'], 0)        
            self.assertEqual(exp.action, Action.HOLD)
            self.assertEqual(exp.reward, -1)
            self.assertEqual(exp.new_state.simulation['money'], 1000)
            self.assertEqual(exp.new_state.simulation['asset'], 0)

            arbiter.set_next_action(Action.BUY)   
            exp = experience_provider.provide_experience()  #(0; 10->11/11)
            self.assertIsNotNone(exp)
            exp, weight = exp
            self.assertEqual(exp.old_state.episode, exp.new_state.episode)
            self.assertEqual(exp.new_state.episode, 0)
            self.assertEqual(exp.old_state.simulation ['money'], 1000)
            self.assertEqual(exp.old_state.simulation['asset'], 0)        
            self.assertEqual(exp.action, Action.BUY)
            self.assertEqual(exp.reward, 2)
            self.assertEqual(exp.new_state.simulation['money'], 0)
            self.assertEqual(exp.new_state.simulation['asset'], 1000 / 10)

            self.assertTrue("position_open" in env)
            self.assertTrue("position" in env)
            self.assertEqual(env['position'], 1000)

            exp = experience_provider.provide_experience()  #(0; 11->12/12)
            self.assertIsNotNone(exp)
            exp, weight = exp
            self.assertEqual(exp.old_state.episode, exp.new_state.episode)
            self.assertEqual(exp.new_state.episode, 0)
            self.assertEqual(exp.old_state.simulation ['money'], 0)
            self.assertEqual(exp.old_state.simulation['asset'], 100)        
            self.assertEqual(exp.action, Action.BUY)
            self.assertEqual(exp.reward, -10)
            self.assertEqual(exp.new_state.simulation['money'], 0)
            self.assertEqual(exp.new_state.simulation['asset'], 100)


            self.assertTrue("position_open" in env)
            self.assertTrue("position" in env)
            self.assertEqual(env['position'], 1000)



            arbiter.set_next_action(Action.HOLD)  
            exp = experience_provider.provide_experience()  #(1; 5->6/10)
            env = experience_provider._reward_processor._working_env

            self.assertFalse("position_open" in env)
            self.assertIsNotNone(exp)
            exp, weight = exp
            self.assertEqual(exp.old_state.episode, exp.new_state.episode)
            self.assertEqual(exp.new_state.episode, 1)
            self.assertEqual(exp.old_state.simulation ['money'], 1000)
            self.assertEqual(exp.old_state.simulation['asset'], 0)        
            self.assertEqual(exp.action, Action.HOLD)
            self.assertEqual(exp.reward, -1)
            self.assertEqual(exp.new_state.simulation['money'], 1000)
            self.assertEqual(exp.new_state.simulation['asset'], 0)


            for i in range(2):
                exp = experience_provider.provide_experience()
                self.assertIsNotNone(exp)
                exp, weight = exp
                self.assertEqual(exp.old_state.episode, exp.new_state.episode)
                self.assertEqual(exp.new_state.episode, 1)

            for j in range(4):
                for i in range(5):
                    exp = experience_provider.provide_experience()
                    self.assertIsNotNone(exp)
                    exp, weight = exp
                    self.assertEqual(exp.old_state.episode, exp.new_state.episode)
                    self.assertEqual(exp.new_state.episode, j+2)
                    
            self.assertIsNone(experience_provider.provide_experience())


    def test_speed(self):
        conf = {
            "process": [
                {"!": "if", "condition": "not {valid_action}", "then": [
                    {"!": "return", "value": "{invalid_action_penalty}"}
                ]},

                {"!": "switch", "condition": "{action}", "cases": [
                    {"value": "buy", "then": [
                        {"!": "store", "name": "position_open"},
                        {"!": "store", "name": "position", "value": "{previous_money}"},
                        {"!": "return", "value": "{buy_reward}"}
                    ]},
                    {"value": "sell", "then": [
                        {"!": "clear", "name": "position_open"},
                        {"!": "return", "value": "{current_money} - {position}"}
                    ]},
                    {"value": "hold", "then": [
                        {"!": "if", "condition": "'position_open' in env", "then":[
                            {"!": "return", "value": "{hold_penalty}"}
                        ], "else": [
                            {"!": "return", "value": "{wait_penalty}"}
                        ]}
                    ]}                    
                ]}
            ],

            "constants": {
                "invalid_action_penalty": -10,
                "buy_reward": 2,
                "hold_penalty": 0,
                "wait_penalty": -1
            }
        }
        
        proc = CommandProcessor(conf)

        def test_proc():
            env = {'valid_action': False,
                          'action': 'buy',
                          'previous_money': 1000,
                          'current_money': 1000}
            proc.reset()
            proc.execute(env)
            env['valid_action'] = True
            proc.execute(env)
            env['action'] = 'hold'
            proc.execute(env)
            env['action'] = 'sell'
            proc.execute(env)

        def py_method(env: dict):
            # implement the same logic as in the CommandProcessor
            if not env['valid_action']:
                return -10
            if env['action'] == 'buy':
                position_open = True
                env['position'] = env['previous_money']
                return 2
            if env['action'] == 'hold':
                if 'position_open' in env:
                    return 0
                return -1
            if env['action'] == 'sell':
                return env['current_money'] - env['position']
            raise ValueError("Invalid action")

        def test_py():
            env = {'valid_action': False,
                          'action': 'buy',
                          'previous_money': 1000,
                          'current_money': 1000}
            py_method(env)
            env['valid_action'] = True
            py_method(env)
            env['action'] = 'hold'
            py_method(env)
            env['action'] = 'sell'
            py_method(env)


        import timeit
        print(timeit.timeit(test_proc, number=10000))
        print(timeit.timeit(test_py, number=10000))


