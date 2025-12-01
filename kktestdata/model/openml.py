import copy
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd
    import numpy as np
    import polars as pl
    import torch
from kktestdata.check import __check_columns
from sklearn.datasets import fetch_openml
from kklogger import set_logger

from ..base import BaseDataset, DatasetMetadata
from ..utils import detect_label_mapping, apply_label_mapping
from ..catalog.openml import OpenMLSpec


LOGGER = set_logger(__name__)


class OpenMLDataset(BaseDataset):
    def _load_pandas(self, strategy: str | None = None) -> "pd.DataFrame":
        LOGGER.info("START")
        meta = self.metadata
        data = fetch_openml(
            name=meta.name,
            version=(meta.source_options or {}).get("version"),
            target_column=None,
            return_X_y=False,
            as_frame=True,
        )
        df = data["data"]
        assert meta.columns_feature is not None
        assert meta.columns_target  is not None
        assert all(isinstance(x, str) and x in df.columns for x in meta.columns_feature)
        columns_target = __check_columns(meta.columns_target)
        df = df[list(meta.columns_feature) + columns_target]
        # auto detect label mapping
        if len(meta.label_mapping_target) == 0:
            if meta.supported_task in ["binary", "multiclass"]:
                dict_label = detect_label_mapping(df[columns_target])
                for _, y in dict_label.items():
                    for a, b in y.items():
                        self.metadata.label_mapping_target[a] = b
        if len(meta.label_mapping_feature) == 0:
            dict_label = detect_label_mapping(df[meta.columns_feature])
            for x, y in dict_label.items():
                self.metadata.label_mapping_feature[x] = copy.deepcopy(y)
        # apply label mapping
        if len(self.metadata.label_mapping_target) > 0:
            if isinstance(list(self.metadata.label_mapping_target.values())[0], dict):
                df = apply_label_mapping(df, self.metadata.label_mapping_target)
            else:
                df = apply_label_mapping(df, {meta.columns_target: self.metadata.label_mapping_target})
        if len(self.metadata.label_mapping_feature) > 0:
            df = apply_label_mapping(df, self.metadata.label_mapping_feature)
        # apply strategy
        if strategy is not None:
            assert strategy in meta.strategy
            df = getattr(self, f"strategy_{strategy}")(df)
        LOGGER.info("END")
        return df
    
    def _load_numpy(self, strategy: str | None = None) -> tuple["np.ndarray", "np.ndarray"]:
        LOGGER.info("START")
        df    = self._load_pandas(strategy=strategy)
        ndf_x = df[self.metadata.columns_feature].to_numpy()
        ndf_y = df[self.metadata.columns_target ].to_numpy()
        LOGGER.info("END")
        return ndf_x, ndf_y
    
    def _load_polars(self, strategy: str | None = None) -> "pl.DataFrame":
        LOGGER.info("START")
        df = self._load_pandas(strategy=strategy)
        df = pl.from_dataframe(df)
        LOGGER.info("END")
        return df

    def _load_torch(self, strategy: str | None = None) -> tuple["torch.Tensor", "torch.Tensor"]:
        LOGGER.info("START")
        ndf_x, ndf_y = self._load_numpy(strategy=strategy)
        ndf_x = torch.from_numpy(ndf_x)
        ndf_y = torch.from_numpy(ndf_y)
        LOGGER.info("END")
        return ndf_x, ndf_y


def build_openml_metadata(spec: OpenMLSpec, strategy: str | list[str] | None=None) -> DatasetMetadata:
    return DatasetMetadata(
        name=spec.name,
        description=spec.description,
        source_type="openml",
        source_options={"version": spec.version,},
        data_type="tabular",
        supported_formats=("numpy", "pandas"),
        supported_task=spec.task,
        columns_target=spec.target,
        columns_feature=spec.features,
        strategy=strategy,
        label_mapping_target={},
        label_mapping_feature={},
        revision="v1.0.0",
    )