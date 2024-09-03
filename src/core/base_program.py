

import logging
from aithena.core.data.assets.asset_manager import AssetManager
from aithena.core.data.normalizer import Normalizer
from aithena.core.nn.dynamic_nn import DynamicNN
from aithena.core.qlearning.dqn_trainer import DQNTrainer
from aithena.core.qlearning.q_arbiter import ArbiterFactory, DeepQFunction
from aithena.core.simulation.sample_provider import SampleProvider
from aithena.core.simulation.state_manager import StateManager
from config_loader import ConfigFiles, ConfigLoader
from evaluation import MetricFactory
from exchange_manager import StateSourcedExchanger
from experience_evaluator import ExperienceEvaluator
from trading_environment import TradingEnvironment
from training_manager import TrainingManager, TrainingReporter


class BaseProgram:


    def __init__(self,
                 config_files: ConfigFiles,
                 use_cuda: bool = True):

        self._training_thread = None

        logging.log(logging.INFO, 'load configs...')
        self._cfg = ConfigLoader(config_files)

        logging.log(logging.INFO, 'load and prepare training data...')
        state_manager = StateManager()
        normalizer = Normalizer()
        asset_manager = AssetManager(self._cfg.assets, False)
        sample_provider = SampleProvider.from_config(
            normalizer, asset_manager, state_manager, self._cfg.input)

        logging.log(logging.INFO, 'prepare agent and trainer...')
        dqn = DynamicNN(self._cfg.nn, self._cfg.input)
        if use_cuda:
            dqn = dqn.cuda()

        arbiter = ArbiterFactory.create(DeepQFunction(dqn),
                                        self._cfg.arbiter.type,
                                        self._cfg.arbiter.params)

        exchanger = StateSourcedExchanger(self._cfg.exchanger)

        env = TradingEnvironment(self._cfg.environment, exchanger)

        trainer = DQNTrainer.from_config(dqn,
                                         self._cfg.training)

        self._trainer = TrainingManager(
            state_manager,
            env,
            arbiter,
            trainer,
            sample_provider,
            self._cfg.training)

        logging.info("ready to train")

        self.evaluator = ExperienceEvaluator.from_cfg(self._cfg.evaluation,
                                                      MetricFactory())
        self.reporter = None

    def start_training_async(self):
        self.reporter = TrainingReporter(self.evaluator)
        self._trainer.start_training_async(self.reporter)

    def is_running(self) -> bool:
        return self._trainer.is_running()
