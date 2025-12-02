from dataclasses import dataclass, asdict
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd
    import numpy as np
    import polars as pl
    import torch
from kklogger import set_logger
from .check import (
    check_source_type, check_data_type, check_strategy, check_supported_formats,
    check_supported_task, check_columns_target, check_columns_feature, check_label_mapping_target,
    check_label_mapping_feature, check_revision, check_cache_root, check_columns_is_null
)
from .utils import to_display


class DatasetError(Exception):
    pass

class UnsupportedFormatError(DatasetError):
    pass

class MissingDependencyError(DatasetError):
    pass

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

    def __init__(self, metadata: DatasetMetadata):
        self.metadata = metadata
        self._validate_metadata()
        self.logger = set_logger(f"{__package__}.{metadata.name}")

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

    def load_data(self, format: str | None = None) -> Any:
        assert format is None or isinstance(format, str) and format
        if format is None:
            format = self.metadata.supported_formats[0]
        assert format in self.metadata.supported_formats
        if format == "pandas":
            return self.load_pandas()
        elif format == "numpy":
            return self.load_numpy()
        elif format == "polars":
            return self.load_polars()
        elif format == "torch":
            return self.load_torch()
        elif format == "dataloader":
            return self.load_dataloader()
        else:
            raise UnsupportedFormatError(f"Unsupported format {format}")
    
    def load_pandas(self) -> "pd.DataFrame":
        pd = _import_pandas()
        if "pandas" in self.metadata.supported_formats:
            ins = self._load_pandas()
            assert isinstance(ins, pd.DataFrame)
            return ins
        else:
            raise UnsupportedFormatError(f"Unsupported format {self.metadata.supported_formats}")
    
    def _load_pandas(self) -> "pd.DataFrame":
        if "pandas" in self.metadata.supported_formats:
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
    
    def load_numpy(self) -> "np.ndarray":
        np = _import_numpy()
        if "numpy" in self.metadata.supported_formats:
            ins = self._load_numpy()
            assert isinstance(ins, (tuple, list)) and len(ins) == 2
            assert all(isinstance(x, np.ndarray) for x in ins)
            return ins
        else:
            raise UnsupportedFormatError(f"Unsupported format {self.metadata.supported_formats}")
    
    def _load_numpy(self) -> "np.ndarray":
        if "numpy" in self.metadata.supported_formats:
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
    
    def load_polars(self) -> "pl.DataFrame":
        pl = _import_polars()
        if "polars" in self.metadata.supported_formats:
            ins = self._load_polars()
            assert isinstance(ins, pl.DataFrame)
            return ins
        else:
            raise UnsupportedFormatError(f"Unsupported format {self.metadata.supported_formats}")
    
    def _load_polars(self) -> "pl.DataFrame":
        if "polars" in self.metadata.supported_formats:
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
    
    def load_torch(self) -> "torch.Tensor":
        torch = _import_torch()
        if "torch" in self.metadata.supported_formats:
            ins = self._load_torch()
            assert isinstance(ins, (tuple, list)) and len(ins) == 2
            assert all(isinstance(x, torch.Tensor) for x in ins)
            return ins
        else:
            raise UnsupportedFormatError(f"Unsupported format {self.metadata.supported_formats}")
    
    def _load_torch(self) -> "torch.Tensor":
        if "torch" in self.metadata.supported_formats:
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
    
    def load_dataloader(self) -> "torch.utils.data.DataLoader":
        torch = _import_torch()
        if "dataloader" in self.metadata.supported_formats:
            ins = self._load_dataloader()
            assert isinstance(ins, torch.utils.data.DataLoader)
            return ins
        else:
            raise UnsupportedFormatError(f"Unsupported format {self.metadata.supported_formats}")
    
    def _load_dataloader(self) -> "torch.utils.data.DataLoader":
        if "dataloader" in self.metadata.supported_formats:
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

# Runtime imports are deferred to keep optional dependencies optional at runtime
def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise MissingDependencyError("pandas is required to load format 'pandas'") from exc
    return pd

def _import_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise MissingDependencyError("numpy is required to load format 'numpy'") from exc
    return np

def _import_polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise MissingDependencyError("polars is required to load format 'polars'") from exc
    return pl

def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise MissingDependencyError("torch is required to load format 'torch' or 'dataloader'") from exc
    return torch

__all__ = [
    "BaseDataset",
    "DatasetMetadata",
    "DatasetError",
    "UnsupportedFormatError",
    "MissingDependencyError",
    "ALLOWED_SOURCE_TYPES",
    "ALLOWED_DATA_TYPES",
    "ALLOWED_TASKS",
    "ALLOWED_FORMATS",
    "ALLOWED_STRATEGIES_BASE",
    "OPTION_STRATEGY_PATTERN",
    "REVISION_PATTERN",
    "to_display",
]
