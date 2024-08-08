from typing import Dict

from core.data.assets.asset_manager import AssetManager
from core.data.assets.asset_provider import AssetProvider
from core.data.assets.asset_source import AssetSource
from core.data.data_provider \
    import ChunkProvider, ChunkType, ContinuousProvider, DataProvider, Sample
from core.data.normalizer import Normalizer
from core.data.technical_indicators.collection import IndicatorCollection
from core.nn.dynamic_nn import DynamicNN
from core.simulation.state_manager import StateManager, StateProvider


class SampleProvider:
    """
    A class that provides samples for simulation.

    Args:
        data_providers (Dict[str, DataProvider]): A dictionary of data
                                                  providers.
    """

    @staticmethod
    def from_config(normalizer: Normalizer,
                    asset_manager: AssetManager,
                    state_manager: StateManager,
                    config: DynamicNN.Config.Input):

        providers = {}

        asset_srcs = []

        for input in config.data:
            if input.type == 'asset':
                input_cfg: AssetManager.Config.Provider = input.config
                df = asset_manager.get_asset_df(input_cfg.asset)
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
                config.input_window)

        return SampleProvider(providers)

    def __init__(self, data_providers: Dict[str, DataProvider]) -> None:

        if data_providers is None or len(data_providers) == 0:
            raise ValueError("Data providers must be provided.")

        self._peek_key = None

        self._chunk_providers = {
            key: provider for key, provider in data_providers.items()
            if isinstance(provider, ChunkProvider)
        }
        self._cont_providers = {
            key: provider for key, provider in data_providers.items()
            if isinstance(provider, ContinuousProvider)
        }

        if len(self._chunk_providers) == 0:
            raise ValueError("At least one chunk provider must be provided.")

        # assert data providers are compatible
        signature = None
        for key in self._chunk_providers:
            sig = self._chunk_providers[key].get_chunk_signature()
            if signature is None:
                signature = sig
                self._peek_key = key
            elif signature != sig:
                raise ValueError("Data providers have incompatible chunk "
                                 "signatures.")

        self._episode_iterators = None  # iterates data chunks
        self._sample_reader = None  # iterates samples within a chunk

    def get_sample_cnt(self, type: ChunkType) -> int:
        return self._chunk_providers[self._peek_key].get_sample_cnt(type)

    def reset(self, type: ChunkType) -> None:
        """
        Reset the sample provider for a new session.

        Args:
            type (ChunkType): The type of data chunk to reset.

        Returns:
            None

        """
        self._episode_iterators = {
            key: self._chunk_providers[key].get_iterator(type)
            for key in self._chunk_providers
        }

        self._cont_providers = {
            key: self._cont_providers[key].get_iterator(type)
            for key in self._cont_providers
        }

        self._next_episode()

    def advance(self) -> Dict[str, Sample]:
        """
        Get the next samples from the sample provider.

        Returns:
            Dict[str, Sample]: A dictionary of samples.

        Raises:
            RuntimeError: If there is no active session.

        """
        if self._sample_reader is None:
            raise RuntimeError("No active session.")
        try:
            return {
                key: next(self._sample_reader[key])
                for key in self._sample_reader.keys()
            } | {key: next(self._cont_providers[key])
                 for key in self._cont_providers.keys()}
        except StopIteration:
            if not self._next_episode():
                raise StopIteration("No more episodes.")
            return self.advance()

    def get_context_samples(self, context: dict) -> Dict[str, Sample]:
        """
        Update the values of the data providers.

        Args:
            samples (Dict[str, Sample]): A dictionary of samples.

        Returns:
            None

        """
        return {
            k: self._cont_providers[k].provide_sample(context)
            for k in self._cont_providers.keys()
        }

    def current_episode_complete(self) -> bool:
        """
        Check if the current episode is complete.

        Returns:
            bool: True if the current episode is complete, False otherwise.

        """
        if self._sample_reader is None:
            raise RuntimeError("No active session.")

        return self._sample_reader[self._peek_key].is_exhausted()

    def _next_episode(self) -> bool:
        """
        Move to the next episode.

        Returns:
            bool: True if there is a next episode, False otherwise.

        """
        self._sample_reader = {
            key: next(self._episode_iterators[key], None)
            for key in self._episode_iterators.keys()
        }
        if self._sample_reader[self._peek_key] is None:
            self._sample_reader = None
            return False
        return True
