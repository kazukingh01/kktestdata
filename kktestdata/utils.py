import pandas as pd
import numpy as np


def detect_label_mapping(df: pd.DataFrame) -> dict[str, int]:
    assert isinstance(df, pd.DataFrame)
    ret_dict = {}
    for col in df.columns:
        if isinstance(df[col].dtypes, pd.CategoricalDtype):
            ret_dict[col] = {x: i for i, x in enumerate(df[col].dtypes.categories)}
        elif df[col].dtype == object:
            try:
                ret_dict[col] = {x: i for i, x in enumerate(np.sort(df[col].unique()))}
            except TypeError as e:
                raise TypeError(f"{e}\n{df[col].unique()}")
    return ret_dict

def apply_label_mapping(df: pd.DataFrame, mapping: dict[str, dict[str, int]]) -> pd.DataFrame:
    assert isinstance(df, pd.DataFrame)
    assert isinstance(mapping, dict) and len(mapping) > 0
    df = df.copy()
    for col, dictwk in mapping.items():
        assert isinstance(dictwk, dict) and len(dictwk) > 0
        df[col] = df[col].map(dictwk).astype(int)
    return df
