from typing import TYPE_CHECKING, Callable, Any
from ..base import BaseDataset, DatasetMetadata
from ..utils import get_dependencies
from ..catalog.huggingface import HFSpec

# import dependencies if it's ready to use
pd, np, pl, torch, load_dataset, hf_hub_download = get_dependencies(["pd", "np", "pl", "torch", "datasets.load_dataset", "huggingface_hub.hf_hub_download"])
if TYPE_CHECKING:
    import pandas as pd
    import numpy as np
    import polars as pl
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download


class HuggingFaceDatasetPandas(BaseDataset):
    def _domain_load_pandas(self, strategy: Callable[Any, Any] | None = None) -> pd.DataFrame:
        self.logger.info("START")
        repo_id     = (self.metadata.source_options or {}).get("repo_id")
        split       = (self.metadata.source_options or {}).get("split", "train")
        load_kwargs = (self.metadata.source_options or {}).get("load_kwargs", {})
        assert isinstance(repo_id, str) and repo_id
        data = load_dataset(repo_id, split=split, **load_kwargs)
        df   = data.to_pandas()
        self.logger.info("END")
        return df
    _domain_load_pandas.is_post_proc = True

    def _domain_load_numpy(self, strategy: Callable[Any, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
        self.logger.info("START")
        df = self._load_pandas(strategy=strategy)
        ndf_x = df[self.metadata.columns_feature].to_numpy()
        ndf_y = df[self.metadata.columns_target].to_numpy()
        self.logger.info("END")
        return ndf_x, ndf_y
    _domain_load_numpy.is_post_proc = False

    def _domain_load_polars(self, strategy: Callable[Any, Any] | None = None) -> pl.DataFrame:
        self.logger.info("START")
        df = self._load_pandas(strategy=strategy)
        df = pl.from_dataframe(df)
        self.logger.info("END")
        return df
    _domain_load_polars.is_post_proc = False

    def _domain_load_torch(self, strategy: Callable[Any, Any] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        self.logger.info("START")
        ndf_x, ndf_y = self._domain_load_numpy(strategy=strategy)
        ndf_x = torch.from_numpy(ndf_x)
        ndf_y = torch.from_numpy(ndf_y)
        self.logger.info("END")
        return ndf_x, ndf_y
    _domain_load_numpy.is_post_proc = False


class HuggingFaceDatasetPolars(BaseDataset):
    def _domain_load_pandas(self, strategy: Callable[Any, Any] | None = None) -> pd.DataFrame:
        self.logger.info("START")
        df = self._load_polars(strategy=strategy)
        df = df.to_pandas()
        self.logger.info("END")
        return df
    _domain_load_pandas.is_post_proc = False

    def _domain_load_numpy(self, strategy: Callable[Any, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
        self.logger.info("START")
        df = self._load_polars(strategy=strategy)
        ndf_x = df[self.metadata.columns_feature].to_numpy()
        ndf_y = df[self.metadata.columns_target ].to_numpy()
        self.logger.info("END")
        return ndf_x, ndf_y
    _domain_load_numpy.is_post_proc = False

    def _domain_load_polars(self, strategy: Callable[Any, Any] | None = None) -> pl.DataFrame:
        self.logger.info("START")
        repo_id  = (self.metadata.source_options or {}).get("repo_id")
        filename = (self.metadata.source_options or {}).get("filename")
        assert isinstance(repo_id,  str) and repo_id
        assert isinstance(filename, str) and filename
        localpath = hf_hub_download(repo_id, repo_type="dataset", filename=filename, token=True)
        df = pl.read_parquet(localpath)
        self.logger.info("END")
        return df
    _domain_load_polars.is_post_proc = True

    def _domain_load_torch(self, strategy: Callable[Any, Any] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        self.logger.info("START")
        ndf_x, ndf_y = self._domain_load_numpy(strategy=strategy)
        ndf_x = torch.from_numpy(ndf_x)
        ndf_y = torch.from_numpy(ndf_y)
        self.logger.info("END")
        return ndf_x, ndf_y
    _domain_load_torch.is_post_proc = False


def build_hf_metadata(spec: HFSpec, strategy: list[str] | None = None) -> DatasetMetadata:
    return DatasetMetadata(
        name=spec.name,
        description=spec.description,
        source_type="huggingface",
        source_options={
            "repo_id": spec.repo_id,
            "split": spec.split,
            "filename": spec.filename,
            "load_kwargs": spec.load_kwargs or {},
        },
        data_type="tabular",
        supported_formats=("numpy", "pandas", "polars", "torch"),
        supported_task=spec.task,
        n_data=spec.n_data,
        n_classes=spec.n_classes,
        columns_target=spec.target,
        columns_feature=spec.features,
        columns_is_null={},
        column_group=spec.group,
        strategy=strategy,
        label_mapping_target={},
        label_mapping_feature={},
        revision="v1.0.0",
    )
