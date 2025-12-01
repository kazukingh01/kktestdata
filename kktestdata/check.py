import re
from typing import Any


ALLOWED_SOURCE_TYPES = {"openml"}
ALLOWED_DATA_TYPES = {"tabular", "image", "language"}
ALLOWED_TASKS = {"binary", "multiclass", "regression", "rank", "multi-regression", "multitask"}
ALLOWED_FORMATS = {"pandas", "numpy", "polars", "torch", "dataloader"}
ALLOWED_STRATEGIES_BASE = {"none", "mean", "median"}
OPTION_STRATEGY_PATTERN = re.compile(r"^option\d+$")
REVISION_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def check_source_type(source_type: str, source_options: Any = None):
    assert isinstance(source_type, str) and source_type
    assert source_type in ALLOWED_SOURCE_TYPES
    if source_options is not None:
        if source_type == "openml":
            assert isinstance(source_options, dict)
            assert all(isinstance(k, str) and k for k in source_options)
        else:
            assert isinstance(source_options, dict)
            assert "version" in source_options
            assert isinstance(source_options["version"], int)

def check_data_type(data_type: str):
    assert isinstance(data_type, str) and data_type
    assert data_type in ALLOWED_DATA_TYPES

def __check_subset(items: tuple[str, ...], allowed_items: set[str]):
    assert isinstance(items, tuple) and len(items) > 0
    assert all(isinstance(x, str) and x for x in items)
    assert set(items).issubset(allowed_items)

def check_supported_formats(supported_formats: tuple[str, ...]):
    __check_subset(supported_formats, ALLOWED_FORMATS)

def check_supported_task(supported_task: str, columns_target: str | list[str] | None = None):
    assert isinstance(supported_task, str) and supported_task
    assert supported_task in ALLOWED_TASKS
    if columns_target is not None:
        columns_target = __check_columns(columns_target, is_allowed_single=True)
        if supported_task == "binary":
            assert len(columns_target) == 1
        elif supported_task == "multiclass":
            assert len(columns_target) == 1
        elif supported_task == "regression":
            assert len(columns_target) == 1
        elif supported_task == "rank":
            assert len(columns_target) == 1
        elif supported_task == "multi-regression":
            assert len(columns_target) > 1
        elif supported_task == "multitask":
            assert len(columns_target) > 1
        else:
            assert False

def check_strategy(strategy: str | list[str] | None, instance: Any = None):
    def _is_valid_strategy(strategy: str) -> bool:
        if strategy in ALLOWED_STRATEGIES_BASE:
            return True
        return bool(OPTION_STRATEGY_PATTERN.match(strategy))
    if strategy is not None:
        strategy_names: list[str]
        if isinstance(strategy, str):
            strategy_names = [strategy]
        else:
            assert isinstance(strategy, (list, tuple)) and len(strategy) > 0
            strategy_names = list(strategy)
        for strategy_name in strategy_names:
            assert isinstance(strategy_name, str) and _is_valid_strategy(strategy_name)
            # ensure strategy handler exists while avoiding AttributeError
            if instance is not None:
                assert getattr(instance, f"strategy_{strategy_name}", None) is not None

def __check_columns(columns: str | int | list[str] | list[int], is_allowed_single: bool = False) -> list[str] | list[int]:
    assert isinstance(is_allowed_single, bool)
    def __check(cols):
        assert isinstance(cols, (list, tuple))
        assert len(cols) > 0
        if isinstance(cols[0], str):
            assert all(isinstance(x, str) for x in cols)
        else:
            assert all(isinstance(x, int) for x in cols)
    if is_allowed_single:
        if isinstance(columns, (list, tuple)):
            __check(columns)
        else:
            assert isinstance(columns, (str, int))
            return [columns, ]
    else:
        __check(columns)
    return list(columns)

def check_columns_target(columns_target: str | list[str]):
    __check_columns(columns_target, is_allowed_single=True)

def check_columns_feature(columns_feature: list[str], columns_target: str | list[str] | None = None):
    __check_columns(columns_feature, is_allowed_single=False)
    if columns_target is not None:
        columns_target = __check_columns(columns_target, is_allowed_single=True)
        assert all(col not in columns_feature for col in columns_target)

def __check_label_mapping(
    label_mapping: dict[str | int, int | dict[str, int]], is_allowed_single: bool = False
) -> dict[str, dict[str, int]]:
    def __check1(x: dict[str, int]):
        assert isinstance(x, dict)
        assert len(x) > 0
        assert all(isinstance(k, str) and isinstance(v, int) for k, v in x.items())
    def __check2(z: dict[str, int]):
        for x, y in z.items():
            assert isinstance(x, str)
            __check1(y)
    assert isinstance(is_allowed_single, bool)
    assert isinstance(label_mapping, dict)
    assert len(label_mapping) > 0
    if is_allowed_single:
        val = list(label_mapping.values())[0]
        if isinstance(val, dict):
            __check2(label_mapping)
        else:
            __check1(label_mapping)
            return {"__dummy__": label_mapping}
    else:
        __check2(label_mapping)
    return label_mapping

def check_label_mapping_target(label_mapping_target: dict[str | int, int | dict[str, int]], columns_target: str | list[str] | None = None):
    label_mapping_target = __check_label_mapping(label_mapping_target, is_allowed_single=True)
    if columns_target is not None:
        columns_target = __check_columns(columns_target, is_allowed_single=True)
        for x, y in label_mapping_target.items():
            if x == "__dummy__":
                assert len(columns_target) == 1
                return True
            assert x in columns_target

def check_label_mapping_feature(label_mapping_feature: dict[str | int, dict[str, int]], columns_feature: list[str] | None = None):
    label_mapping_feature = __check_label_mapping(label_mapping_feature, is_allowed_single=False)
    if columns_feature is not None:
        columns_feature = __check_columns(columns_feature, is_allowed_single=False)
        for x, _ in label_mapping_feature.items():
            assert x in columns_feature

def check_revision(revision: str):
    assert isinstance(revision, str) and revision
    assert REVISION_PATTERN.match(revision)

def check_cache_root(cache_root: str):
    assert isinstance(cache_root, str) and cache_root
