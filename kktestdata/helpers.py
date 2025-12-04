from typing import Any, TYPE_CHECKING
from kklogger import set_logger

from .utils import get_dependencies
from .check import (
    check_split_type, check_split_consistency_dataframe, check_split_consistency_array,
    check_supported_task, check_random_seed, TYPE_DATAFRAME, TYPE_ARRAY, CLASS_DATAFRAME, CLASS_ARRAY
)
from .error import UnsupportedError


RANDOM_SEED = 42
LOGGER = set_logger(__name__)


# import dependencies if it's ready to use
pd, np, pl, torch, train_test_split = get_dependencies(["pd", "np", "pl", "torch", "sklearn.model_selection.train_test_split"])
if TYPE_CHECKING:
    import pandas as pd
    import numpy as np
    import polars as pl
    import torch
    from sklearn.model_selection import train_test_split


def train_test_split_any(
    *inputs: Any, test_size: float, inputs_type: Any = None, stratify: Any = None, seed: int = RANDOM_SEED
) -> tuple[Any, ...]:
    LOGGER.info("START")
    assert isinstance(inputs, tuple) and len(inputs) in [1, 2]
    assert isinstance(test_size, float) and 0 < test_size < 1
    assert isinstance(inputs_type, type) and issubclass(inputs_type, (CLASS_DATAFRAME, CLASS_ARRAY))
    check_random_seed(seed)
    if inputs_type == np.ndarray:
        assert all(isinstance(x, np.ndarray) for x in inputs)
        assert len(inputs) == 2
        assert inputs[0].shape[0] == inputs[1].shape[0]
        if stratify is not None:
            assert isinstance(stratify, np.ndarray) and stratify.shape[0] == inputs[0].shape[0]
    elif inputs_type == torch.Tensor:
        assert all(isinstance(x, torch.Tensor) for x in inputs)
        assert len(inputs) == 2
        assert inputs[0].shape[0] == inputs[1].shape[0]
        if stratify is not None:
            assert isinstance(stratify, torch.Tensor) and stratify.shape[0] == inputs[0].shape[0]
            stratify = stratify.numpy()
    elif inputs_type == pd.DataFrame:
        assert all(isinstance(x, pd.DataFrame) for x in inputs)
        assert len(inputs) == 1
        if stratify is not None:
            assert isinstance(stratify, pd.Series) and stratify.shape[0] == inputs[0].shape[0]
            stratify = stratify.to_numpy()
    elif inputs_type == pl.DataFrame:
        assert all(isinstance(x, pl.DataFrame) for x in inputs)
        assert len(inputs) == 1
        if stratify is not None:
            assert isinstance(stratify, pl.Series) and stratify.shape[0] == inputs[0].shape[0]
            stratify = stratify.to_numpy()
    else:
        raise ValueError(f"Invalid inputs type: {inputs_type}")

    ret_value = None
    indexes   = np.arange(inputs[0].shape[0], dtype=int)
    indexes_train, indexes_test = train_test_split(indexes, test_size=test_size, random_state=seed, stratify=stratify)
    if inputs_type == np.ndarray:
        ret_value = (inputs[0][indexes_train], inputs[1][indexes_train], inputs[0][indexes_test], inputs[1][indexes_test], )
    elif inputs_type == torch.Tensor:
        ret_value = (inputs[0][indexes_train], inputs[1][indexes_train], inputs[0][indexes_test], inputs[1][indexes_test], )
    elif inputs_type == pd.DataFrame:
        ret_value = (inputs[0].iloc[indexes_train], inputs[0].iloc[indexes_test], )
    elif inputs_type == pl.DataFrame:
        ret_value = (inputs[0][indexes_train], inputs[0][indexes_test], )
    LOGGER.info("END")
    return *ret_value, indexes_train

def split_by_mode_task(
    *inputs: Any, check_type: TYPE_DATAFRAME | TYPE_ARRAY = None, mode: str = "train",
    task: str = "binary", test_size: float = 0.2, valid_size: float = None, stratify: Any = None, seed: int = RANDOM_SEED
) -> tuple[Any, ...]:
    LOGGER.info("START")
    if check_type in CLASS_DATAFRAME:
        inputs = check_split_consistency_dataframe(inputs, check_type=check_type)
    else:
        inputs = check_split_consistency_array(inputs, check_type=check_type)
    check_split_type(mode)
    check_supported_task(task)
    check_random_seed(seed)
    assert isinstance(test_size,  float) and 0 < test_size < 1
    if not task in ("binary", "multiclass", "rank") and stratify is not None:
        raise UnsupportedError(f"Unsupported stratify when task is not binary, multiclass, or rank: {task}")
    if check_type in CLASS_DATAFRAME:
        assert len(inputs) == 1
    else:
        assert len(inputs) == 2
    ret_value = None
    if mode == "train":
        ret_value = inputs
    elif mode == "test":
        if check_type in CLASS_DATAFRAME:
            data_train, data_test, _ = train_test_split_any(inputs[0], test_size=test_size, inputs_type=check_type, stratify=stratify, seed=seed)
            ret_value = (data_train, data_test, )
        else:
            data_train_X, data_train_y, data_test_X, data_test_y, _ = train_test_split_any(
                inputs[0], inputs[1], test_size=test_size, inputs_type=check_type, stratify=stratify, seed=seed
            )
            ret_value = (data_train_X, data_train_y, data_test_X, data_test_y, )
    elif mode == "valid":
        if valid_size is None:
            valid_size = test_size
            LOGGER.warning(f"valid_size is not set, using test_size: {test_size}")
        assert isinstance(valid_size, float) and 0 < valid_size < 1
        if check_type in CLASS_DATAFRAME:
            data_train, data_test, indexes = train_test_split_any(inputs[0], test_size=test_size, inputs_type=check_type, stratify=stratify, seed=seed)
            if stratify is not None:
                if check_type == pd.DataFrame:
                    stratify = stratify.iloc[indexes]
                else:
                    stratify = stratify[indexes]
            data_train, data_valid, _ = train_test_split_any(
                data_train, test_size=((inputs[0].shape[0] * valid_size) / data_train.shape[0]), 
                inputs_type=check_type, stratify=stratify, seed=seed
            )
            ret_value = (data_train, data_valid, data_test, )
        else:
            data_train_X, data_train_y, data_test_X, data_test_y, indexes = train_test_split_any(
                inputs[0], inputs[1], test_size=test_size, inputs_type=check_type, stratify=stratify, seed=seed
            )
            if stratify is not None:
                stratify = stratify[indexes]
            data_train_X, data_train_y, data_valid_X, data_valid_y, _ = train_test_split_any(
                data_train_X, data_train_y, test_size=((inputs[0].shape[0] * valid_size) / data_train_X.shape[0]), 
                inputs_type=check_type, stratify=stratify, seed=seed
            )
            ret_value = (data_train_X, data_train_y, data_valid_X, data_valid_y, data_test_X, data_test_y, )
    LOGGER.info("END")
    return ret_value
