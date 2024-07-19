

from dataclasses import dataclass
import json
import pandas as pd
import torch
from program.asset_source import AssetSource
from core.data.data_provider\
      import ChunkIterator, ChunkProvider, ChunkReader, ChunkType, Sample


class AssetProvider(ChunkProvider):

    @dataclass
    class ChunkRange:
        start_ix: int
        length: int

    def __init__(self,
                 source: AssetSource,
                 context_columns: list[str],
                 tensor_columns: list[str],
                 window_size: int) -> None:

        self._window_size = window_size

        data = source.get_data([
            AssetSource.DataFrameRequirement(
                key='chunks',
                columns=['chunk', 'chunk_type'],
                normalize=False),
            AssetSource.DataFrameRequirement(
                key='context',
                columns=context_columns,
                normalize=False),
            AssetSource.DataFrameRequirement(
                key='tensor',
                columns=tensor_columns,
                normalize=True)
        ])

        chunk_id_col = data['chunks']['chunk']
        chunk_type_col = data['chunks']['chunk_type']

        self._context = data['context']

        self._tensor = torch.Tensor(data['tensor'].values.astype('float32'))

        # determine chunk ranges as tuples [start_ix, length]
        chunk_ids = chunk_id_col[chunk_id_col != -1].unique()
        self._chunk_ranges = []
        for id in chunk_ids:
            sub_frame = self._context[chunk_id_col == id]
            self._chunk_ranges.append(
                AssetProvider.ChunkRange(
                    start_ix=sub_frame.index[0],
                    length=len(sub_frame)))

        # collect ids according to chunktype
        self._chunk_ids = {
            ChunkType.TRAINING: [],
            ChunkType.VALIDATION: [],
            ChunkType.TEST: []
        }
        for r in self._chunk_ranges:
            type = ChunkType.from_int(chunk_type_col[r.start_ix])
            self._chunk_ids[type].append(chunk_id_col[r.start_ix])

    # returns an iterator over the data chunks of the specified type
    def get_iterator(self, chunk_type: ChunkType) -> ChunkIterator:
        return AssetChunkIterator(self, chunk_type)

    def get_chunk_cnt(self, chunk_type: ChunkType) -> int:
        return len(self._chunk_ids[chunk_type])

    def get_chunk_signature(self) -> str:
        res = {}
        for type in ChunkType:
            res[type] = []
            for id in self._chunk_ids[type]:
                res[type].append(self._chunk_ranges[id].length)
        return json.dumps(res)


class AssetChunkIterator(ChunkIterator):
    def __init__(self,
                 provider: AssetProvider,
                 chunk_type: ChunkType) -> None:
        super().__init__()
        self._provider = provider

        self._ids = provider._chunk_ids[chunk_type]

        self._chunk_type = chunk_type
        self._ix = 0

    def __len__(self) -> int:
        return len(self._ids)

    """Returns the next chunk of data and the original data."""
    def __next__(self) -> 'AssetChunkReader':
        if self._ix == len(self._ids):
            raise StopIteration()

        chunk_id = self._ids[self._ix]
        r = self._provider._chunk_ranges[chunk_id]
        ix_start = r.start_ix
        ix_end = r.start_ix + r.length

        self._ix += 1

        tensor = self._provider._tensor[ix_start:ix_end]
        context = self._provider._context.iloc[ix_start:ix_end]

        return AssetChunkReader(tensor,
                                context,
                                self._provider._window_size)


class AssetChunkReader(ChunkReader):
    def __init__(self,
                 tensor: torch.Tensor,
                 context: pd.DataFrame,
                 window_size: int) -> None:
        super().__init__()
        self._tensor = tensor
        self._context = context
        self._window_size = window_size

        self._ix = 0

    def __len__(self) -> int:
        return len(self._tensor) - self._window_size + 1

    def __next__(self) -> Sample:
        if self.is_exausted():
            raise StopIteration()

        ix_start = self._ix
        ix_end = self._ix + self._window_size

        self._ix += 1

        return Sample(self._tensor[ix_start:ix_end],
                      self._context.iloc[ix_end-1])

    def is_exausted(self) -> bool:
        return self._ix + self._window_size > len(self._tensor)
