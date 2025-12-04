import copy, random
from dataclasses import dataclass, asdict
from typing import Any, TYPE_CHECKING, final
from kklogger import set_logger
from .check import (
    check_source_type, check_data_type, check_strategy, check_supported_formats,
    check_supported_task, check_columns_target, check_columns_feature, check_label_mapping_target,
    check_label_mapping_feature, check_revision, check_cache_root, check_columns_is_null,
    check_split_type, check_split_consistency, check_columns, check_column_group
)
from .error import (
    DatasetError, UnsupportedFormatError, MissingDependencyError
)
from .utils import to_display, get_dependencies, detect_label_mapping, apply_label_mapping
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
DICT_TYPE_NAME     = {
    "pandas": pd.DataFrame,
    "numpy":  np.ndarray,
    "polars": pl.DataFrame,
    "torch":  torch.Tensor,
    "dataloader": DataLoader,
}

@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    description: str
    source_type: str
    data_type: str  # tabular | image | language
    supported_formats: tuple[str, ...] # pandas, numpy, polars, torch, dataloader
    supported_task: str  # binary | multiclass | regression | rank
    n_data: int | None = None
    n_classes: int | None = None
    columns_target: str | list[str] | None = None
    columns_feature: list[str] | None = None
    columns_is_null: dict[str | int, bool] | None = None
    column_group: str | None = None # group for ranking
    strategy: list[str] | None = None
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
        check_supported_task(self.metadata.supported_task, columns_target=self.metadata.columns_target, column_group=self.metadata.column_group)
        check_strategy(self.metadata.strategy, instance=self)
        if self.metadata.columns_target is not None:
            check_columns_target( self.metadata.columns_target)
        if self.metadata.columns_feature is not None:
            check_columns_feature(self.metadata.columns_feature, columns_target=self.metadata.columns_target)
        if self.metadata.columns_is_null is not None and len(self.metadata.columns_is_null) > 0:
            check_columns_is_null(self.metadata.columns_is_null, columns_feature=self.metadata.columns_feature)
        if self.metadata.column_group is not None:
            check_column_group(self.metadata.column_group, columns_feature=self.metadata.columns_feature)
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
        "n_data", "n_classes", "n_target", "n_features", "n_null_columns"
    ]) -> str:
        return to_display(self.to_dict(list_keys=list_keys))

    def load_data(
        self, format: str | None = None, split_type: str = "train", test_size: float = 0.2, valid_size: float = None,
        strategy: str | None = None
    ) -> Any:
        self.logger.info("START")
        assert format is None or isinstance(format, str) and format
        assert strategy is None or isinstance(strategy, str) and strategy
        if format is None:
            format = self.metadata.supported_formats[0]
        assert format in self.metadata.supported_formats
        check_split_type(split_type)
        self.set_random_seed()
        if strategy is None and self.metadata.strategy is not None and len(self.metadata.strategy) > 0:
            strategy = self.metadata.strategy[0]
        if format == "pandas":
            ins = self._load_pandas(strategy=strategy)
        elif format == "numpy":
            ins = self._load_numpy(strategy=strategy)
        elif format == "polars":
            ins = self._load_polars(strategy=strategy)
        elif format == "torch":
            ins = self._load_torch(strategy=strategy)
        elif format == "dataloader":
            ins = self._load_dataloader(strategy=strategy)
        else:
            raise UnsupportedFormatError(f"Unsupported format {format}")
        # split
        if self.metadata.supported_task in ("binary", "multiclass", "rank"):
            if format == "pandas":
                stratify = ins[self.metadata.columns_target]
            elif format == "numpy":
                stratify = ins[1]
            elif format == "polars":
                stratify = ins[self.metadata.columns_target]
            elif format == "torch":
                stratify = ins[1]
            elif format == "dataloader":
                stratify = None
        else:
            stratify = None
        if self.metadata.supported_task == "rank" and self.metadata.column_group is not None:
            if format == "pandas":
                groups = ins[self.metadata.column_group].to_numpy()
            elif format == "numpy":
                groups = ins[0][:, np.argmax(self.metadata.column_group == np.array(self.metadata.columns_feature))]
            elif format == "polars":
                groups = ins[self.metadata.column_group].to_numpy()
            elif format == "torch":
                groups = ins[0][:, np.argmax(self.metadata.column_group == np.array(self.metadata.columns_feature))]
            elif format == "dataloader":
                groups = None
            if groups is not None:
                dictwk = {x: i for i, x in enumerate(np.sort(np.unique(groups)))}
                groups = np.vectorize(lambda x: dictwk[x])(groups)
        else:
            groups = None
        check_type = DICT_TYPE_NAME[format]
        ins = check_split_consistency(ins, check_type=check_type)
        ins = split_by_mode_task(
            *ins, check_type=check_type, mode=split_type, task=self.metadata.supported_task,
            test_size=test_size, valid_size=valid_size, stratify=stratify, groups=groups, seed=self.seed
        )
        if isinstance(ins, (tuple, list)) and len(ins) == 1:
            ins = ins[0]
        self.logger.info("END")
        return ins

    @final
    def _load_pandas(self, strategy: str | None = None) -> TypeLoadPandas:
        if "pandas" in self.metadata.supported_formats:
            self.logger.info("START")
            df = self._domain_load_pandas(strategy=strategy)
            check_for_pandas(self.metadata, df)
            df = df.loc[:, list(self.metadata.columns_feature) + check_columns(self.metadata.columns_target, is_allowed_single=True)]
            # apply strategy
            if strategy is not None:
                assert strategy in self.metadata.strategy
                self.logger.info("Applying strategy %s", strategy)
                df = getattr(self, f"strategy_{strategy}")(df)
                assert isinstance(df, pd.DataFrame)
            # check columns is null
            self.metadata = create_columns_is_null(self.metadata, df)
            # auto detect label mapping
            self.metadata = create_label_mapping_from_dataframe(self.metadata, df)
            # apply label mapping
            df = apply_label_mapping_to_dataframe(self.metadata, df)
            self.logger.info("END")
            return df
        else:
            raise UnsupportedFormatError(f"Unsupported format {self.metadata.supported_formats}")
    
    def _domain_load_pandas(self) -> pd.DataFrame:
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
    
    @final
    def _load_numpy(self, strategy: str | None = None) -> TypeLoadNumpy:
        if "numpy" in self.metadata.supported_formats:
            self.logger.info("START")
            ins = self._domain_load_numpy(strategy=strategy)
            self.logger.info("END")
            return ins
        else:
            raise UnsupportedFormatError(f"Unsupported format {self.metadata.supported_formats}")
    
    def _domain_load_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
    
    @final
    def _load_polars(self, strategy: str | None = None) -> TypeLoadPolars:
        if "polars" in self.metadata.supported_formats:
            self.logger.info("START")
            ins = self._domain_load_polars()
            self.logger.info("END")
            return ins
        else:
            raise UnsupportedFormatError(f"Unsupported format {self.metadata.supported_formats}")
    
    def _domain_load_polars(self) -> pl.DataFrame:
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
    
    @final
    def _load_torch(self, strategy: str | None = None) -> TypeLoadTorch:
        if "torch" in self.metadata.supported_formats:
            self.logger.info("START")
            ins = self._domain_load_torch(strategy=strategy)
            self.logger.info("END")
            return ins
        else:
            raise UnsupportedFormatError(f"Unsupported format {self.metadata.supported_formats}")
    
    def _domain_load_torch(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
    
    @final
    def _load_dataloader(self, strategy: str | None = None) -> TypeLoadDataloader:
        if "dataloader" in self.metadata.supported_formats:
            self.logger.info("START")
            ins = self._domain_load_dataloader(strategy=strategy)
            assert isinstance(ins, DataLoader)
            self.logger.info("END")
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

def check_for_pandas(meta: DatasetMetadata, df: pd.DataFrame):
    assert isinstance(meta, DatasetMetadata)
    assert isinstance(df, pd.DataFrame)
    assert meta.columns_feature is not None
    assert meta.columns_target  is not None
    assert all(isinstance(x, str) and x in df.columns for x in meta.columns_feature), \
        f"columns_feature: {meta.columns_feature} not in {df.columns.tolist()}"

def create_label_mapping_from_dataframe(meta: DatasetMetadata, df: pd.DataFrame) -> DatasetMetadata:
    assert isinstance(meta, DatasetMetadata)
    assert isinstance(df, pd.DataFrame)    
    columns_target = [meta.columns_target, ] if isinstance(meta.columns_target, str) else meta.columns_target
    meta = copy.deepcopy(meta)
    # auto detect label mapping
    if len(meta.label_mapping_target) == 0:
        if meta.supported_task in ["binary", "multiclass"]:
            dict_label = detect_label_mapping(df[columns_target])
            for _, y in dict_label.items():
                for a, b in y.items():
                    meta.label_mapping_target[a] = b
    if len(meta.label_mapping_feature) == 0:
        dict_label = detect_label_mapping(df[meta.columns_feature])
        for x, y in dict_label.items():
            meta.label_mapping_feature[x] = copy.deepcopy(y)
    return meta

def apply_label_mapping_to_dataframe(meta: DatasetMetadata, df: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(meta, DatasetMetadata)
    assert isinstance(df, pd.DataFrame)
    meta = copy.deepcopy(meta)
    if len(meta.label_mapping_target) > 0:
        if isinstance(list(meta.label_mapping_target.values())[0], dict):
            df = apply_label_mapping(df, meta.label_mapping_target)
        else:
            assert isinstance(meta.columns_target, str)
            df = apply_label_mapping(df, {meta.columns_target: meta.label_mapping_target})
    if len(meta.label_mapping_feature) > 0:
        df = apply_label_mapping(df, meta.label_mapping_feature)
    return df

def create_columns_is_null(meta: DatasetMetadata, df: pd.DataFrame) -> DatasetMetadata:
    assert isinstance(meta, DatasetMetadata)
    assert isinstance(df, pd.DataFrame)
    meta = copy.deepcopy(meta)
    for col in meta.columns_feature:
        meta.columns_is_null[col] = bool(df[col].isnull().any())
    return meta

__all__ = [
    "BaseDataset",
    "DatasetMetadata",
    "DatasetError",
    "UnsupportedFormatError",
    "MissingDependencyError",
    "to_display",
    "to_dict",
]
