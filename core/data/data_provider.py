
from abc import abstractmethod
from dataclasses import dataclass
import enum
import json
from typing import Self
import torch


class ChunkType(enum.Enum):
    TRAINING = "tr"
    VALIDATION = "val"
    TEST = "test"

    def as_int(self) -> int:
        if self == ChunkType.TRAINING:
            return 0
        elif self == ChunkType.VALIDATION:
            return 1
        elif self == ChunkType.TEST:
            return 2
        else:
            raise ValueError(f"Chunk type {self} not supported.")

    @staticmethod
    def from_int(i: int) -> Self:
        if i == 0:
            return ChunkType.TRAINING
        elif i == 1:
            return ChunkType.VALIDATION
        elif i == 2:
            return ChunkType.TEST
        else:
            raise ValueError(f"Chunk type with value {i} not supported.")

    @staticmethod
    def from_str(s: str) -> Self:
        if s == "tr":
            return ChunkType.TRAINING
        elif s == "val":
            return ChunkType.VALIDATION
        elif s == "test":
            return ChunkType.TEST
        else:
            raise ValueError(f"Chunk type {s} not supported.")


@dataclass
class Sample:
    tensor: torch.Tensor
    context: dict

    def __hash__(self) -> int:
        if self.tensor is None:
            h1 = 0
        else:
            h1 = hash(str(self.tensor))

        if self.context is None:
            h2 = 0
        else:
            h2 = hash(json.dumps(self.context))

        return hash(h1+h2)

    def __eq__(self, other: Self) -> bool:
        if other is None:
            return False

        if self.tensor is None or other.tensor is None:
            return self.tensor == other.tensor \
                and self.context == other.context

        return torch.equal(self.tensor, other.tensor) \
            and self.context == other.context


class ChunkReader:
    def __iter__(self) -> Self:
        return self

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __next__(self) -> Sample:
        pass

    @abstractmethod
    def is_exhausted(self) -> bool:
        pass


class ChunkIterator:
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __next__(self) -> ChunkReader:
        pass

    def __iter__(self) -> Self:
        return self

    @abstractmethod
    def __len__(self) -> int:
        pass


class DataProvider:
    @abstractmethod
    def get_iterator(self, chunk_type: ChunkType):
        pass


class ContinuousProvider(DataProvider):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_iterator(self, chunk_type: ChunkType) -> Self:
        pass

    @abstractmethod
    def __next__(self) -> Sample:
        pass

    @abstractmethod
    def provide_sample(self, context: dict):
        pass


class ChunkProvider(DataProvider):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_iterator(self, chunk_type: ChunkType) -> ChunkIterator:
        pass

    @abstractmethod
    def get_chunk_signature(self) -> str:
        pass
