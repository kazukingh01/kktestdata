import re
import pandas as pd
import numpy as np
from typing import Any


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

