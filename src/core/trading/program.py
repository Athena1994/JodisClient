
from aithena.core.data.assets.asset_manager import AssetManager
from aithena.core.data.assets.asset_provider import AssetProvider
from aithena.core.data.assets.asset_source import AssetSource
from aithena.core.data.normalizer import Normalizer
from aithena.core.data.technical_indicators.collection import IndicatorCollection
from aithena.core.nn.dynamic_nn import DynamicNN
from aithena.core.qlearning.dqn_trainer import DQNTrainer
from aithena.core.qlearning.q_arbiter import DeepQFunction, EpsilonGreedyArbiter
from aithena.core.simulation.sample_provider import SampleProvider
from aithena.core.simulation.state_manager import StateManager, StateProvider
from config_loader import ConfigFiles, ConfigLoader
from exchange_manager import StateSourcedExchanger
from trading_environment import TradingEnvironment
from training_manager import TrainingManager


class Program:

    def __init__(self,
                 config_files: ConfigFiles):
        print('loading configs...')
        self._config = ConfigLoader(config_files)

        print("loading asset data...")
        state_manager = StateManager()
        sample_provider = self._init_sample_provider(state_manager)

        print('preparing training...')
        self._trainer = self._init_training_manager(state_manager,
                                                    sample_provider)

    def train(self):
        print("start training...")
        self._trainer._train_epoch()
        print("finished!")

    def _init_sample_provider(self, state_manager: StateManager):
        am = AssetManager(self._config.assets, False)

        normalizer = Normalizer()

        return AssetProvider.from_config(am, normalizer, self._config.input)

        providers = {}

        asset_srcs = []

        for input in self._config.input.data:
            if input.type == 'asset':
                input_cfg: AssetManager.Config.Provider = input.config
                df = am.get_asset_df(input_cfg.asset)
                asset_srcs.append((AssetSource(df,
                                               normalizer,
                                               input_cfg.normalizer),
                                   input))
            if input.type == 'state':
                providers[input.key] = StateProvider(state_manager,
                                                     normalizer,
                                                     input.config)

        def get_columns(cfg: AssetManager.Config.Provider) -> list[str]:
            return cfg.include \
                  + [IndicatorCollection.get_from_cfg(i).get_unique_id()
                     for i in cfg.indicators]

        for src, data_cfg in asset_srcs:
            providers[data_cfg.key] = AssetProvider(
                src,
                data_cfg.config.include,
                get_columns(data_cfg.config),
                self._config.input.input_window)

        return SampleProvider(providers)

    def _init_training_manager(self,
                               state_manager: StateManager,
                               sample_provider: SampleProvider):

        dqn = DynamicNN(self._config.nn, self._config.input).cuda()

        arbiter = EpsilonGreedyArbiter(
            DeepQFunction(dqn),
            self._config.training.exploration.sigma)

        exchanger = StateSourcedExchanger(state_manager, self._config.exchanger)

        env = TradingEnvironment(self._config.environment, exchanger)

        trainer = DQNTrainer.from_config(dqn,
                                         self._config.training)

        return TrainingManager(state_manager,
                               env,
                               arbiter,
                               trainer,
                               sample_provider,
                               self._config.training)
