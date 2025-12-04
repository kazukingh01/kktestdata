from typing import TYPE_CHECKING
from ...model.huggingface import HuggingFaceDataset, build_hf_metadata
from ...catalog.huggingface import SPEC_BY_NAME
from ...utils import get_dependencies

pd = get_dependencies(["pd"])
if TYPE_CHECKING:
    import pandas as pd


DATASET_NAME = "boatrace_2020_2021"


class Dataset(HuggingFaceDataset):
    metadata = build_hf_metadata(SPEC_BY_NAME[DATASET_NAME], strategy=["v1"])

    def strategy_v1(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Drop 'rank_web' is null")
        columns = df.columns.tolist()
        df["is_null_rank_web"] = df["rank_web"].isnull()
        dfwk = df.groupby("race_id").agg({"is_null_rank_web": "max"}).reset_index().rename(columns={"is_null_rank_web": "is_null"})
        df = pd.merge(df, dfwk, how="left", on="race_id")
        df = df.loc[df["is_null"] == False, columns]
        df["rank_web"] = df["rank_web"].astype(int)
        return df.copy()