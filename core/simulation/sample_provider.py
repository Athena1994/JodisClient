from typing import Dict

from core.data.data_provider import ChunkProvider, ChunkType, ContinuousProvider, DataProvider, Sample

class SampleProvider:
    """
    A class that provides samples for simulation.

    Args:
        data_providers (Dict[str, DataProvider]): A dictionary of data providers.
    """

    def __init__(self, data_providers: Dict[str, DataProvider]) -> None:

        self._chunk_provider_dict = {}
        self._cont_providers = {}
        self._peek_key = None

        self._chunk_provider_dict = {
            key: provider for key, provider in data_providers.items()
            if isinstance(provider, ChunkProvider)
        }
        self._cont_providers = {
            key: provider for key, provider in data_providers.items()
            if isinstance(provider, ContinuousProvider)
        }

        # assert data providers are compatible
        signature = None
        for key, provider in self._cont_providers.values():
            sig = provider.get_chunk_signature()
            if signature is None:
                signature = sig
                self._peek_key = key
            elif signature != sig:
                raise ValueError("Data providers have incompatible chunk signatures.")

        self._episode_iterators = None  # iterates data chunks 
        self._sample_iterators = None  # iterates samples within a chunk

    def reset(self, type: ChunkType) -> None:
        """
        Reset the sample provider for a new session.

        Args:
            type (ChunkType): The type of data chunk to reset.

        Returns:
            None

        """
        self._episode_iterators = {
            key: self._data_provider[key].get_iterator(type)
            for key in self._data_providers.keys()
        }
        self._next_episode()

    def _next_episode(self) -> bool:
        """
        Move to the next episode.

        Returns:
            bool: True if there is a next episode, False otherwise.

        """
        self._sample_iterators = {
            key: next(self._episode_iterators[key], None)
            for key in self._episode_iterators.keys()
        }
        if self._sample_iterators[self._peek_key] is None:
            self._sample_iterators = None
            return False
        return True

    def get_next_samples(self) -> Dict[str, Sample]:
        """
        Get the next samples from the sample provider.

        Returns:
            Dict[str, Sample]: A dictionary of samples.

        Raises:
            RuntimeError: If there is no active session.

        """
        if self._sample_iterators is None:
            raise RuntimeError("No active session.")
        try:
            return {
                key: next(self._sample_iterators[key])
                for key in self._sample_iterators.keys()
            } | {key: next(self._cont_providers[key])
                 for key in self._cont_providers.keys()}
        except StopIteration:
            if not self._next_episode():
                raise StopIteration("No more episodes.")
            return self.get_next_sample()

    def update_values(self, samples: Dict[str, Sample]):
        """
        Update the values of the data providers.

        Args:
            samples (Dict[str, Sample]): A dictionary of samples.

        Returns:
            None

        """
        updated_values = {
            self._cont_providers[k].update_sample(samples[k]) 
            for k in self._cont_providers.keys()
        }
        samples.update(updated_values)

    def current_episode_complete(self) -> bool:
        """
        Check if the current episode is complete.

        Returns:
            bool: True if the current episode is complete, False otherwise.

        """
        return self._sample_iterators[self._peek_key].is_exausted()