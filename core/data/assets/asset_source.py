
from dataclasses import dataclass
from typing import Dict

import pandas as pd

from core.data.normalizer import Normalizer


class AssetSource:

    @dataclass
    class DataFrameRequirement:
        key: str
        columns: list[str]
        normalize: bool

    def __init__(self,
                 df: pd.DataFrame,
                 normalizer: Normalizer,
                 normalizer_conf: Normalizer.Config) -> None:

        self._df = df.copy()
        self._normalizer = normalizer
        self._normalizer_conf = normalizer_conf

        normalizer.prepare(self._df, self._normalizer_conf)

    def get_data(self, requirements: list[DataFrameRequirement]) \
            -> Dict[str, pd.DataFrame]:

        res = {}

        for req in requirements:
            # assert columns are present
            missing_cols = [col for col in req.columns
                            if col not in self._df]
            if len(missing_cols) > 0:
                raise ValueError(f"Columns {missing_cols} not found in data.")

            res[req.key] = self._df[req.columns].copy()
            if req.normalize:
                res[req.key] \
                    = self._normalizer.normalize_df(res[req.key],
                                                    self._normalizer_conf)
        return res
