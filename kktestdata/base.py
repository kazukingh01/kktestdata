import copy, random
from dataclasses import dataclass, asdict
from typing import Any, TYPE_CHECKING, final
from kklogger import set_logger
from .check import (
    check_source_type, check_data_type, check_strategy, check_supported_formats,
    check_supported_task, check_columns_target, check_columns_feature, check_label_mapping_target,
    check_label_mapping_feature, check_revision, check_cache_root, check_columns_is_null,
    check_split_type, check_split_consistency_dataframe, check_split_consistency_array
)
from .error import (
    DatasetError, UnsupportedFormatError, MissingDependencyError
)
from .utils import to_display, get_dependencies
from .helpers import split_by_mode_task

# import dependencies if it's ready to use
pd, np, pl, torch, DataLoader = get_dependencies(["pd", "np", "pl", "torch", "torch.utils.data.DataLoader"])
if TYPE_CHECKING:
    import pandas as pd
    import numpy as np
    import polars as pl
    import torch
    from torch.utils.data import DataLoader


TypeLoadPandas     = pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
TypeLoadNumpy      = tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
TypeLoadPolars     = pl.DataFrame | tuple[pl.DataFrame, pl.DataFrame] | tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
TypeLoadTorch      = tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
TypeLoadDataloader = DataLoader | tuple[DataLoader, DataLoader] | tuple[DataLoader, DataLoader, DataLoader]

@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    description: str
    source_type: str
    data_type: str  # tabular | image | language
    supported_formats: tuple[str, ...] # pandas, numpy, polars, torch, dataloader
    supported_task: str  # binary | multiclass | regression | rank
    n_data: int | None = None
    columns_target: str | list[str] | None = None
    columns_feature: list[str] | None = None
    columns_is_null: dict[str | int, bool] | None = None
    strategy: str | list[str] | None = None
    label_mapping_target:  dict[str | int, int | dict[str, int]] = None # {feature_name | index: {label: index}}
    label_mapping_feature: dict[str | int,       dict[str, int]] = None # {feature_name | index: {label: index}}
    source_options: dict[str, Any] | None = None
    revision: str = "v1.0.0"
    cache_root: str = f"~/.cache/{__package__}"

class BaseDataset:
    metadata: DatasetMetadata

    def __init__(self, metadata: DatasetMetadata, seed: int = 42):
        assert isinstance(seed, int) and seed >= 0
        self.metadata = copy.deepcopy(metadata)
        self._validate_metadata()
        self.seed = seed
        self.set_random_seed()
        self.logger = set_logger(f"{__package__}.{metadata.name}")
    
    def set_random_seed(self):
        random.seed(self.seed)
        if getattr(np, "random", None) is not None:
            np.random.seed(self.seed)
        if getattr(torch, "manual_seed", None) is not None:
            torch.manual_seed(self.seed)
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        if getattr(torch, "backends", None) is not None:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @final
    def _validate_metadata(self):
        assert isinstance(self.metadata.name,        str) and self.metadata.name
        assert isinstance(self.metadata.description, str) and self.metadata.description
        check_source_type(self.metadata.source_type, source_options=self.metadata.source_options)
        check_data_type(self.metadata.data_type)
        check_supported_formats(self.metadata.supported_formats)
        check_supported_task(self.metadata.supported_task, columns_target=self.metadata.columns_target)
        check_strategy(self.metadata.strategy, instance=self)
        if self.metadata.columns_target is not None:
            check_columns_target( self.metadata.columns_target)
        if self.metadata.columns_feature is not None:
            check_columns_feature(self.metadata.columns_feature, columns_target=self.metadata.columns_target)
        if self.metadata.columns_is_null is not None and len(self.metadata.columns_is_null) > 0:
            check_columns_is_null(self.metadata.columns_is_null, columns_feature=self.metadata.columns_feature)
        if self.metadata.label_mapping_target is not None and len(self.metadata.label_mapping_target) > 0:
            check_label_mapping_target( self.metadata.label_mapping_target,  columns_target =self.metadata.columns_target)
        if self.metadata.label_mapping_feature is not None and len(self.metadata.label_mapping_feature) > 0:
            check_label_mapping_feature(self.metadata.label_mapping_feature, columns_feature=self.metadata.columns_feature)
        check_revision(self.metadata.revision)
        check_cache_root(self.metadata.cache_root)

    def to_dict(self, list_keys: list[str] | None = None) -> dict[str, Any]:
        return to_dict(self.metadata, list_keys=list_keys)

    def to_display(self, list_keys: list[str] | None = [
        "name", "source_type", "data_type", "supported_formats", "supported_task", 
        "n_data", "n_target", "n_features", "n_null_columns"
    ]) -> str:
        return to_display(self.to_dict(list_keys=list_keys))

    def load_data(self, format: str | None = None, split_type: str = "train", test_size: float = 0.2, valid_size: float = None) -> Any:
        assert format is None or isinstance(format, str) and format
        if format is None:
            format = self.metadata.supported_formats[0]
        assert format in self.metadata.supported_formats
        check_split_type(split_type)
        self.set_random_seed()
        if format == "pandas":
            return self._load_pandas(split_type=split_type, test_size=test_size, valid_size=valid_size)
        elif format == "numpy":
            return self._load_numpy(split_type=split_type, test_size=test_size, valid_size=valid_size)
        elif format == "polars":
            return self._load_polars(split_type=split_type, test_size=test_size, valid_size=valid_size)
        elif format == "torch":
            return self._load_torch(split_type=split_type, test_size=test_size, valid_size=valid_size)
        elif format == "dataloader":
            return self._load_dataloader()
        else:
            raise UnsupportedFormatError(f"Unsupported format {format}")
    
    @final
    def _load_pandas(self, split_type: str = "train", test_size: float = None, valid_size: float = None) -> TypeLoadPandas:
        if "pandas" in self.metadata.supported_formats:
            ins = self._domain_load_pandas()
            check_split_consistency_dataframe(ins, check_type=pd.DataFrame)
            if self.metadata.supported_task in ("binary", "multiclass", "rank"):
                stratify = ins[self.metadata.columns_target]
            else:
                stratify = None
            ins = split_by_mode_task(
                ins, check_type=pd.DataFrame, mode=split_type, task=self.metadata.supported_task,
                test_size=test_size, valid_size=valid_size, stratify=stratify, seed=self.seed
            )
            return ins
        else:
            raise UnsupportedFormatError(f"Unsupported format {self.metadata.supported_formats}")
    
    def _domain_load_pandas(self) -> pd.DataFrame:
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
    
    @final
    def _load_numpy(self, split_type: str = "train", test_size: float = None, valid_size: float = None) -> TypeLoadNumpy:
        if "numpy" in self.metadata.supported_formats:
            ins = self._domain_load_numpy()
            check_split_consistency_array(ins, check_type=np.ndarray)
            if self.metadata.supported_task in ("binary", "multiclass", "rank"):
                stratify = ins[1]
            else:
                stratify = None
            ins = split_by_mode_task(
                *ins, check_type=np.ndarray, mode=split_type, task=self.metadata.supported_task,
                test_size=test_size, valid_size=valid_size, stratify=stratify, seed=self.seed
            )
            return ins
        else:
            raise UnsupportedFormatError(f"Unsupported format {self.metadata.supported_formats}")
    
    def _domain_load_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
    
    @final
    def _load_polars(self, split_type: str = "train", test_size: float = None, valid_size: float = None) -> TypeLoadPolars:
        if "polars" in self.metadata.supported_formats:
            ins = self._domain_load_polars()
            check_split_consistency_dataframe(ins, check_type=pl.DataFrame)
            if self.metadata.supported_task in ("binary", "multiclass", "rank"):
                stratify = ins[self.metadata.columns_target]
            else:
                stratify = None
            ins = split_by_mode_task(
                ins, check_type=pl.DataFrame, mode=split_type, task=self.metadata.supported_task,
                test_size=test_size, valid_size=valid_size, stratify=stratify, seed=self.seed
            )
            return ins
        else:
            raise UnsupportedFormatError(f"Unsupported format {self.metadata.supported_formats}")
    
    def _domain_load_polars(self) -> pl.DataFrame:
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
    
    @final
    def _load_torch(self, split_type: str = "train", test_size: float = None, valid_size: float = None) -> TypeLoadTorch:
        if "torch" in self.metadata.supported_formats:
            ins = self._domain_load_torch()
            check_split_consistency_array(ins, check_type=torch.Tensor)
            if self.metadata.supported_task in ("binary", "multiclass", "rank"):
                stratify = ins[1]
            else:
                stratify = None
            ins = split_by_mode_task(
                *ins, check_type=torch.Tensor, mode=split_type, task=self.metadata.supported_task,
                test_size=test_size, valid_size=valid_size, stratify=stratify, seed=self.seed
            )
            return ins
        else:
            raise UnsupportedFormatError(f"Unsupported format {self.metadata.supported_formats}")
    
    def _domain_load_torch(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
    
    @final
    def _load_dataloader(self) -> TypeLoadDataloader:
        if "dataloader" in self.metadata.supported_formats:
            ins = self._domain_load_dataloader()
            assert isinstance(ins, DataLoader)
            return ins
        else:
            raise UnsupportedFormatError(f"Unsupported format {self.metadata.supported_formats}")
    
    def _domain_load_dataloader(self) -> DataLoader:
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
    
def to_dict(meta: DatasetMetadata, list_keys: list[str] | None = None) -> dict[str, Any]:
    assert isinstance(meta, DatasetMetadata)
    meta = asdict(meta)
    if meta.get("columns_target") is not None and not isinstance(meta.get("columns_target"), (tuple, list)):
        meta["columns_target"] = [meta["columns_target"], ]
    meta["n_features"]     = len(meta["columns_feature"]) if meta.get("columns_feature") is not None else None
    meta["n_target"]       = len(meta["columns_target"])  if meta.get("columns_target")  is not None else None
    meta["n_null_columns"] = sum(list(meta.get("columns_is_null", {}).values())) if meta.get("columns_is_null") is not None else None
    if list_keys is None:
        list_keys = list(meta.keys())
    else:
        assert (isinstance(list_keys, list) and all(isinstance(k, str) for k in list_keys))
        assert all(k in meta.keys() for k in list_keys)
    return {k: meta.get(k) for k in list_keys}


__all__ = [
    "BaseDataset",
    "DatasetMetadata",
    "DatasetError",
    "UnsupportedFormatError",
    "MissingDependencyError",
    "to_display",
    "to_dict",
]
