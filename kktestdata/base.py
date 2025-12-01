import re
from dataclasses import dataclass, asdict
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
    supported_tasks: tuple[str, ...]  # binary | multiclass | regression | rank
    columns_target: str | list[str] | None = None
    columns_feature: list[str] | None = None
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
        def _is_valid_strategy(strategy: str) -> bool:
            if strategy in ALLOWED_STRATEGIES_BASE:
                return True
            return bool(OPTION_STRATEGY_PATTERN.match(strategy))
        meta = self.metadata
        assert isinstance(meta.name, str) and meta.name
        assert isinstance(meta.description, str) and meta.description
        assert meta.source_type in ALLOWED_SOURCE_TYPES
        if meta.source_type == "openml":
            assert meta.source_options is None or isinstance(meta.source_options, dict)
            if isinstance(meta.source_options, dict):
                assert all(isinstance(k, str) and k for k in meta.source_options)
        else:
            assert isinstance(meta.source_options, dict)
        assert meta.data_type in ALLOWED_DATA_TYPES
        assert isinstance(meta.supported_formats, tuple) and len(meta.supported_formats) > 0
        assert all(isinstance(x, str) and x for x in meta.supported_formats)
        assert set(meta.supported_formats).issubset(ALLOWED_FORMATS)
        assert isinstance(meta.supported_tasks, tuple) and len(meta.supported_tasks) > 0
        assert set(meta.supported_tasks).issubset(ALLOWED_TASKS)
        # strategy
        if meta.strategy is not None:
            strategy_names: list[str]
            if isinstance(meta.strategy, str):
                strategy_names = [meta.strategy]
            else:
                assert isinstance(meta.strategy, (list, tuple)) and len(meta.strategy) > 0
                strategy_names = list(meta.strategy)
            for strategy_name in strategy_names:
                assert isinstance(strategy_name, str) and _is_valid_strategy(strategy_name)
                # ensure strategy handler exists while avoiding AttributeError
                assert getattr(self, f"strategy_{strategy_name}", None) is not None
        # columns_target
        assert meta.columns_target  is None or (isinstance(meta.columns_target,  (str, list, tuple)))
        if isinstance(meta.columns_target, (list, tuple)):
            assert len(meta.columns_target) > 0
            assert all(isinstance(x, str) and x for x in meta.columns_target)
        # columns_feature
        assert meta.columns_feature is None or (isinstance(meta.columns_feature, (list, tuple)))
        if meta.columns_feature is not None:
            assert len(meta.columns_feature) > 0
            assert all(isinstance(x, str) and x for x in meta.columns_feature)
            if meta.columns_target is not None:
                if isinstance(meta.columns_target, str):
                    assert meta.columns_target not in meta.columns_feature
                else:
                    assert all(x not in meta.columns_target for x in meta.columns_feature)
        # label_mapping_target
        assert isinstance(meta.label_mapping_target, dict)
        if len(meta.label_mapping_target) > 0:
            # {label: label_index} or {feature_name | index: {label: label_index}}
            for x, y in meta.label_mapping_target.items():
                assert isinstance(x, (str, int))
                if isinstance(x, int):
                    assert isinstance(y, dict) # pattern: {target_index: {label: label_index}}
                    for a, b in y.items():
                        assert isinstance(a, str) and isinstance(b, int)
                else:
                    assert isinstance(y, (int, dict))
                    if isinstance(y, int):
                        assert isinstance(y, int) # pattern: {label: label_index}
                    else:
                        assert x in meta.columns_target
                        for a, b in y.items():
                            assert isinstance(a, str) and isinstance(b, int)
        # label_mapping_feature
        assert isinstance(meta.label_mapping_feature, dict)
        if len(meta.label_mapping_feature) > 0:
            # pattern: {feature_name | index: {label: label_index}}
            for x, y in meta.label_mapping_feature.items():
                assert isinstance(x, (str, int))
                if isinstance(x, str):
                    assert x in meta.columns_feature
                assert isinstance(y, dict)
                for a, b in y.items():
                    assert isinstance(a, str) and isinstance(b, int)
        assert isinstance(meta.revision, str) and REVISION_PATTERN.match(meta.revision)
        assert isinstance(meta.cache_root, str) and meta.cache_root

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
