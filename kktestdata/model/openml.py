from typing import TYPE_CHECKING
from ..base import BaseDataset, DatasetMetadata
from ..utils import get_dependencies
from ..catalog.openml import OpenMLSpec

# import dependencies if it's ready to use
pd, np, pl, torch, fetch_openml = get_dependencies(["pd", "np", "pl", "torch", "sklearn.datasets.fetch_openml"])
if TYPE_CHECKING:
    import pandas as pd
    import numpy as np
    import polars as pl
    import torch
    from sklearn.datasets import fetch_openml


class OpenMLDataset(BaseDataset):
    def _domain_load_pandas(self, strategy: str | None = None) -> pd.DataFrame:
        self.logger.info("START")
        data = fetch_openml(
            name=self.metadata.name,
            version=(self.metadata.source_options or {}).get("version"),
            target_column=None,
            return_X_y=False,
            as_frame=True,
        )
        df = data["data"]
        self.logger.info("END")
        return df
    
    def _domain_load_numpy(self, strategy: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        self.logger.info("START")
        df    = self._load_pandas(strategy=strategy)
        ndf_x = df[self.metadata.columns_feature].to_numpy()
        ndf_y = df[self.metadata.columns_target ].to_numpy()
        self.logger.info("END")
        return ndf_x, ndf_y
    
    def _domain_load_polars(self, strategy: str | None = None) -> pl.DataFrame:
        self.logger.info("START")
        df = self._load_pandas(strategy=strategy)
        df = pl.from_dataframe(df)
        self.logger.info("END")
        return df

    def _domain_load_torch(self, strategy: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        self.logger.info("START")
        ndf_x, ndf_y = self._domain_load_numpy(strategy=strategy)
        ndf_x = torch.from_numpy(ndf_x)
        ndf_y = torch.from_numpy(ndf_y)
        self.logger.info("END")
        return ndf_x, ndf_y
    
    def strategy_v1(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Drop the classes which has low number of samples")
        n       = df.shape[0]
        se      = df.groupby(self.metadata.columns_target, observed=True).size()
        classes = se.index[(se >= (n * 0.0001)) & (se >= 3)].tolist() # over 0.01% or >= 3 samples
        df      = df.loc[df[self.metadata.columns_target].isin(classes), :].copy()
        if isinstance(df[self.metadata.columns_target].dtypes, pd.CategoricalDtype):
            df[self.metadata.columns_target] = df[self.metadata.columns_target].astype(pd.CategoricalDtype(categories=classes, ordered=True))
        from dataclasses import asdict
        meta    = asdict(self.metadata)
        meta["n_data"]    = df.shape[0]
        meta["n_classes"] = len(classes)
        self.metadata = DatasetMetadata(**meta)
        return df

def build_openml_metadata(spec: OpenMLSpec, strategy: list[str] | None=None) -> DatasetMetadata:
    return DatasetMetadata(
        name=spec.name,
        description=spec.description,
        source_type="openml",
        source_options={"version": spec.version,},
        data_type="tabular",
        supported_formats=("numpy", "pandas", "polars", "torch"),
        supported_task=spec.task,
        n_data=spec.n_data,
        n_classes=spec.n_classes,
        columns_target=spec.target,
        columns_feature=spec.features,
        columns_is_null={},
        strategy=strategy,
        label_mapping_target={},
        label_mapping_feature={},
        revision="v1.0.0",
    )
