import re
from importlib import import_module
from typing import Any, TYPE_CHECKING
from kklogger import set_logger


LOGGER = set_logger(__name__)
DEPENDENCIES = ["pd", "np", "pl", "torch", "cv", "sklearn"]


class DummyDataframe:
    class DataFrame:
        pass
class DummyTensor:
    class Tensor:
        pass
class DummyNdarray:
    class ndarray:
        pass

def get_dependencies(list_dependencies: list[str] = ["pd"]) -> tuple[Any, ...]:
    assert isinstance(list_dependencies, list) and len(list_dependencies) > 0
    for mod in list_dependencies:
        if mod.find(".") >= 0:
            module_top = mod.split(".")[0]
            assert module_top in DEPENDENCIES, f"Invalid dependency: {mod}"
        else:
            assert mod in DEPENDENCIES, f"Invalid dependency: {mod}"
    list_ret = []
    for mod in list_dependencies:
        if mod == "pd":
            try:
                import pandas as pd
                list_ret.append(pd)
            except ImportError:
                LOGGER.warning(f"pandas is not installed")
                list_ret.append(DummyDataframe())
        elif mod == "np":
            try:
                import numpy as np
                list_ret.append(np)
            except ImportError:
                LOGGER.warning(f"numpy is not installed")
                list_ret.append(DummyNdarray())
        elif mod == "pl":
            try:
                import polars as pl
                list_ret.append(pl)
            except ImportError:
                LOGGER.warning(f"polars is not installed")
                list_ret.append(DummyDataframe())
        elif mod == "torch":
            try:
                import torch
                list_ret.append(torch)
            except ImportError:
                LOGGER.warning(f"torch is not installed")
                list_ret.append(DummyTensor())
        elif mod == "cv":
            try:
                import cv2
                list_ret.append(cv2)
            except ImportError:
                LOGGER.warning(f"cv2 is not installed")
                list_ret.append(None)
        elif mod == "sklearn":
            try:
                import sklearn
                list_ret.append(sklearn)
            except ImportError:
                LOGGER.warning(f"scikit-learn is not installed")
                list_ret.append(None)
        else:
            try:
                modules  = mod.split(".")
                funcname = modules[-1]
                modules  = ".".join(modules[:-1])
                module   = import_module(modules)
                list_ret.append(getattr(module, funcname))
            except ImportError:
                LOGGER.warning(f"{mod} is not installed")
                list_ret.append(None)
    return tuple(list_ret)

# import dependencies if it's ready to use
pd, np = get_dependencies(["pd", "np"])
if TYPE_CHECKING:
    import pandas as pd
    import numpy as np


def detect_label_mapping(df: pd.DataFrame) -> dict[str, int]:
    assert isinstance(df, pd.DataFrame)
    ret_dict = {}
    for col in df.columns:
        if isinstance(df[col].dtypes, pd.CategoricalDtype):
            ret_dict[col] = {x: i for i, x in enumerate(df[col].dtypes.categories)}
        elif df[col].dtype == object:
            cols  = df[col].unique()
            regex = re.compile("^(<|>|)[0-9]+$")
            if all([regex.match(x) for x in cols if (not isinstance(x, float) or (isinstance(x, float) and not np.isnan(x)))]):
                num_first, num_last = None, None
                for x in cols:
                    if not isinstance(x, float) and re.compile("^<[0-9]+$").match(x):
                        if num_first is None:
                            num_first = int(x[1:])
                        else:
                            raise ValueError(f"Invalid type: {cols}")                        
                for x in cols:
                    if not isinstance(x, float) and re.compile("^>[0-9]+$").match(x):
                        if num_last is None:
                            num_last = int(x[1:])
                        else:
                            raise ValueError(f"Invalid type: {cols}")
                assert num_first is not None
                assert num_last  is not None
                s_cat = pd.Categorical(cols, categories=(
                    [f"<{num_first}"] + [str(x) for x in range(num_first, num_last + 1)] + [f">{num_last}"]
                ), ordered=True)
                ret_dict[col] = {x: i for i, x in enumerate(s_cat.categories)}
            else:
                try:
                    ret_dict[col] = {x: i for i, x in enumerate(np.sort(cols))}
                except TypeError as e:
                    raise TypeError(f"{e}\n{df[col].unique()}")
    return ret_dict

def apply_label_mapping(df: pd.DataFrame, mapping: dict[str, dict[str, int]], fillna: int | float = -1) -> pd.DataFrame:
    assert isinstance(df, pd.DataFrame)
    assert isinstance(mapping, dict) and len(mapping) > 0
    assert isinstance(fillna, (int, float))
    df = df.copy()
    for col, dictwk in mapping.items():
        assert isinstance(dictwk, dict) and len(dictwk) > 0
        if isinstance(fillna, int):
            df[col] = df[col].map(dictwk).astype(float).fillna(fillna).astype(int) # first .astype(float) is for converting categoical to float
        else:
            df[col] = df[col].map(dictwk).astype(float).fillna(fillna).astype(float)
    return df

def to_display(dictwk: dict[str, Any]) -> str:
    w = max(len(k) for k in dictwk.keys()) # max length of key
    lines = ["("]
    for k, v in dictwk.items():
        lines.append(f"  {k.ljust(w)} = {v!r},")
    lines.append(")")
    return "\n".join(lines)
