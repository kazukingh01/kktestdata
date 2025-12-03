import copy
from typing import TYPE_CHECKING
from kklogger import set_logger

from ..base import BaseDataset, DatasetMetadata
from ..check import check_columns
from ..utils import detect_label_mapping, apply_label_mapping, get_dependencies
from ..catalog.openml import OpenMLSpec

# import dependencies if it's ready to use
pd, np, pl, torch, fetch_openml = get_dependencies(["pd", "np", "pl", "torch", "sklearn.datasets.fetch_openml"])
if TYPE_CHECKING:
    import pandas as pd
    import numpy as np
    import polars as pl
    import torch
    from sklearn.datasets import fetch_openml


LOGGER = set_logger(__name__)


class OpenMLDataset(BaseDataset):
    def _domain_load_pandas(self, strategy: str | None = None) -> pd.DataFrame:
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
        assert all(isinstance(x, str) and x in df.columns for x in meta.columns_feature), \
            f"columns_feature: {meta.columns_feature} not in {df.columns.tolist()}"
        columns_target = check_columns(meta.columns_target, is_allowed_single=True)
        df = df[list(meta.columns_feature) + columns_target]
        # check columns is null
        for col in meta.columns_feature:
            meta.columns_is_null[col] = bool(df[col].isnull().any())
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
        LOGGER.info(
            "Fetched dataset %s rows=%s cols=%s target=%s",
            meta.name, df.shape[0], df.shape[1], columns_target,
        )
        LOGGER.info("END")
        return df
    
    def _domain_load_numpy(self, strategy: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        LOGGER.info("START")
        df    = self._domain_load_pandas(strategy=strategy)
        ndf_x = df[self.metadata.columns_feature].to_numpy()
        ndf_y = df[self.metadata.columns_target ].to_numpy()
        LOGGER.info("END")
        return ndf_x, ndf_y
    
    def _domain_load_polars(self, strategy: str | None = None) -> pl.DataFrame:
        LOGGER.info("START")
        df = self._domain_load_pandas(strategy=strategy)
        df = pl.from_dataframe(df)
        LOGGER.info("END")
        return df

    def _domain_load_torch(self, strategy: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        LOGGER.info("START")
        ndf_x, ndf_y = self._domain_load_numpy(strategy=strategy)
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
        supported_formats=("numpy", "pandas", "polars", "torch"),
        supported_task=spec.task,
        n_data=spec.n_data,
        columns_target=spec.target,
        columns_feature=spec.features,
        columns_is_null={},
        strategy=strategy,
        label_mapping_target={},
        label_mapping_feature={},
        revision="v1.0.0",
    )
