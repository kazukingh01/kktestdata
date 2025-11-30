import re
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd
    import numpy as np
    import polars as pl
    import torch
from kklogger import set_logger


ALLOWED_SOURCE_TYPES = {"openml"}
ALLOWED_DATA_TYPES = {"tabular", "image", "language"}
ALLOWED_TASKS = {"binary", "multiclass", "regression", "rank"}
ALLOWED_FORMATS = {"pandas", "numpy", "polars", "torch", "dataloader"}
ALLOWED_STRATEGIES_BASE = {"none", "mean", "median"}
OPTION_STRATEGY_PATTERN = re.compile(r"^option\d+$")
REVISION_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class DatasetError(Exception):
    pass


class DatasetNotFoundError(DatasetError):
    pass


class UnsupportedFormatError(DatasetError):
    pass


class UnsupportedTargetModeError(DatasetError):
    pass


class MissingDependencyError(DatasetError):
    pass


class CacheUnavailableError(DatasetError):
    pass


class SplitConfigurationError(DatasetError):
    pass


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    description: str
    source_type: str
    source_options: dict[str, Any]
    data_type: str  # tabular | image | language
    supported_formats: tuple[str, ...]
    supported_tasks: tuple[str, ...]  # binary | multiclass | regression | rank
    columns_target: str | int | list[str | int]
    columns_feature: tuple[str, ...]
    strategy: str | list[str] | None = None
    label_mapping: dict[str, int] | None = None
    revision: str = "v1.0.0"
    cache_root: str = f"~/.cache/{__package__}"


class BaseDataset:
    metadata: DatasetMetadata

    def __init__(self, metadata: DatasetMetadata):
        self.metadata = metadata
        self._validate_metadata()
        self.logger = set_logger(f"{__package__}.{metadata.name}")

    def _validate_metadata(self):
        meta = self.metadata
        assert isinstance(meta.name, str) and meta.name
        assert isinstance(meta.description, str) and meta.description
        assert meta.source_type in ALLOWED_SOURCE_TYPES
        assert isinstance(meta.source_options, dict)
        if meta.source_type == "openml":
            assert "url" in meta.source_options
            assert isinstance(meta.source_options["url"], str) and meta.source_options["url"]
        assert meta.data_type in ALLOWED_DATA_TYPES
        assert isinstance(meta.supported_formats, tuple) and len(meta.supported_formats) > 0
        assert all(isinstance(x, str) and x for x in meta.supported_formats)
        assert set(meta.supported_formats).issubset(ALLOWED_FORMATS)
        assert isinstance(meta.supported_tasks, tuple) and len(meta.supported_tasks) > 0
        assert set(meta.supported_tasks).issubset(ALLOWED_TASKS)
        if meta.strategy is not None:
            if isinstance(meta.strategy, str):
                assert _is_valid_strategy(meta.strategy)
                assert getattr(self, f"strategy_{meta.strategy}") is not None
            else:
                assert isinstance(meta.strategy, (list, tuple)) and len(meta.strategy) > 0
                for x in meta.strategy:
                    assert isinstance(x, str) and _is_valid_strategy(x)
                    assert getattr(self, f"strategy_{meta.strategy}") is not None
        assert isinstance(meta.columns_target, (str, int, list))
        assert isinstance(meta.columns_feature, tuple) and len(meta.columns_feature) > 0
        assert all(isinstance(x, str) and x for x in meta.columns_feature)
        if isinstance(meta.columns_target, (str, int)):
            assert meta.columns_target not in meta.columns_feature
        if isinstance(meta.columns_target, list):
            for col in meta.columns_target:
                assert isinstance(col, (str, int))
                assert col not in meta.columns_feature
        assert meta.label_mapping is None or isinstance(meta.label_mapping, dict)
        assert isinstance(meta.revision, str) and REVISION_PATTERN.match(meta.revision)
        assert isinstance(meta.cache_root, str) and meta.cache_root
    
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
        if self.metadata.supported_formats.contains("pandas"):
            ins = self._load_pandas()
            assert isinstance(ins, pd.DataFrame)
            return ins
        else:
            raise UnsupportedFormatError(f"Unsupported format {self.metadata.supported_formats}")
    
    def _load_pandas(self) -> "pd.DataFrame":
        if self.metadata.supported_formats.contains("pandas"):
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
    
    def load_numpy(self) -> "np.ndarray":
        if self.metadata.supported_formats.contains("numpy"):
            ins = self._load_numpy()
            assert isinstance(ins, tuple | list) and len(ins) == 2
            assert all(isinstance(x, np.ndarray) for x in ins)
            return ins
        else:
            raise UnsupportedFormatError(f"Unsupported format {self.metadata.supported_formats}")
    
    def _load_numpy(self) -> "np.ndarray":
        if self.metadata.supported_formats.contains("numpy"):
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
    
    def load_polars(self) -> "pl.DataFrame":
        if self.metadata.supported_formats.contains("polars"):
            ins = self._load_polars()
            assert isinstance(ins, pl.DataFrame)
            return ins
        else:
            raise UnsupportedFormatError(f"Unsupported format {self.metadata.supported_formats}")
    
    def _load_polars(self) -> "pl.DataFrame":
        if self.metadata.supported_formats.contains("polars"):
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
    
    def load_torch(self) -> "torch.Tensor":
        if self.metadata.supported_formats.contains("torch"):
            ins = self._load_torch()
            assert isinstance(ins, tuple | list) and len(ins) == 2
            assert all(isinstance(x, torch.Tensor) for x in ins)
            return ins
        else:
            raise UnsupportedFormatError(f"Unsupported format {self.metadata.supported_formats}")
    
    def _load_torch(self) -> "torch.Tensor":
        if self.metadata.supported_formats.contains("torch"):
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
    
    def load_dataloader(self) -> "torch.utils.data.DataLoader":
        if self.metadata.supported_formats.contains("dataloader"):
            ins = self._load_dataloader()
            assert isinstance(ins, torch.utils.data.DataLoader)
            return ins
        else:
            raise UnsupportedFormatError(f"Unsupported format {self.metadata.supported_formats}")
    
    def _load_dataloader(self) -> "torch.utils.data.DataLoader":
        if self.metadata.supported_formats.contains("dataloader"):
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
    
def _is_valid_strategy(strategy: str) -> bool:
    if strategy in ALLOWED_STRATEGIES_BASE:
        return True
    return bool(OPTION_STRATEGY_PATTERN.match(strategy))

__all__ = [
    "BaseDataset",
    "DatasetMetadata",
    "DatasetError",
    "DatasetNotFoundError",
    "UnsupportedFormatError",
    "UnsupportedTargetModeError",
    "MissingDependencyError",
    "CacheUnavailableError",
    "SplitConfigurationError",
    "get_logger",
    "ALLOWED_SOURCE_TYPES",
    "ALLOWED_DATA_TYPES",
    "ALLOWED_TASKS",
    "ALLOWED_FORMATS",
    "ALLOWED_STRATEGIES_BASE",
    "OPTION_STRATEGY_PATTERN",
    "REVISION_PATTERN",
]
