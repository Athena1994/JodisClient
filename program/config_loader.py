from dataclasses import dataclass
import json
import os
from core.data.assets.asset_manager import AssetManager
from core.nn.dynamic_nn import DynamicNN
from core.qlearning.dqn_trainer import DQNTrainer
from core.qlearning.q_arbiter import ExplorationArbiter
from program.exchange_manager import StateSourcedExchanger
from program.experience_evaluator import ExperienceEvaluator
from program.trading_environment import TradingEnvironment
from utils.config_utils import assert_fields_in_dict


@dataclass
class ConfigFiles:
    agent: str
    data: str
    training: str
    simulation: str
    evaluation: str

    @staticmethod
    def from_dict(d: dict):
        assert_fields_in_dict(d, ['agent', 'data', 'training', 'simulation',
                                  'evaluation'])
        return ConfigFiles(
            d['agent'],
            d['data'],
            d['training'],
            d['simulation'],
            d['evaluation']
        )

    def with_base_path(self, base_path: str):
        return ConfigFiles(
            os.path.join(base_path, self.agent),
            os.path.join(base_path, self.data),
            os.path.join(base_path, self.training),
            os.path.join(base_path, self.simulation),
            os.path.join(base_path, self.evaluation)
        )


class ConfigLoader:

    @staticmethod
    def _load_file(path: str):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"File {path} not found.")
            raise Exception(f"File {path} not found.")

    def __init__(self, files: ConfigFiles):

        agent_cfg = self._load_file(files.agent)
        data_cfg = self._load_file(files.data)
        sim_cfg = self._load_file(files.simulation)
        training_cfg = self._load_file(files.training)
        evaluation_cfg = self._load_file(files.evaluation)

        self.input = DynamicNN.Config.Input.from_dict(agent_cfg['input'])
        self.nn = DynamicNN.Config.from_dict(agent_cfg['nn'])

        self.assets = AssetManager.Config.from_dict(data_cfg)

        self.environment \
            = TradingEnvironment.Config.from_dict(sim_cfg['environment'])
        self.exchanger \
            = StateSourcedExchanger.Config.from_dict(sim_cfg['exchanger'])

        self.training = DQNTrainer.Config.from_dict(training_cfg)
        self.evaluation \
            = ExperienceEvaluator.Config.from_dict(evaluation_cfg['evaluation'])
        self.arbiter \
            = ExplorationArbiter.Config.from_dict(training_cfg['exploration'])
