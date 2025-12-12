from typing import TYPE_CHECKING
from ...model.openml import OpenMLDataset
from ...utils import get_dependencies
from ...catalog.openml import SPEC_BY_NAME
from ...model.openml import build_openml_metadata

pd = get_dependencies(["pd"])
if TYPE_CHECKING:
    import pandas as pd


DATASET_NAME = "KDDCup99"


class Dataset(OpenMLDataset):
    metadata = build_openml_metadata(SPEC_BY_NAME[DATASET_NAME], strategy=["v1"])

    def strategy_v1(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Drop the classes which has low number of samples")
        n       = df.shape[0]
        se      = df.groupby(self.metadata.columns_target, observed=True).size()
        classes = se.index[(se >= (n * 0.0001)) & (se >= 3)].tolist() # over 0.01% or >= 3 samples
        return df.loc[df[self.metadata.columns_target].isin(classes), :].copy()