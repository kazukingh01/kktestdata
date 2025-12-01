import re
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
    check_label_mapping_feature, check_revision, check_cache_root
)

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
        if self.metadata.label_mapping_target is not None:
            check_label_mapping_target( self.metadata.label_mapping_target,  columns_target =self.metadata.columns_target)
        if self.metadata.label_mapping_feature is not None:
            check_label_mapping_feature(self.metadata.label_mapping_feature, columns_feature=self.metadata.columns_feature)
        check_revision(self.metadata.revision)
        check_cache_root(self.metadata.cache_root)

    def to_display(self) -> str:
        d = asdict(self.metadata)
        w = max(len(k) for k in d) # max length of key
        lines = [f"{self.__class__.__name__}("]
        for k, v in d.items():
            lines.append(f"  {k.ljust(w)} = {v!r},")
        lines.append(")")
        return "\n".join(lines)

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
]
