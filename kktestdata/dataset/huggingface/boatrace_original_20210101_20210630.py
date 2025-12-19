from typing import TYPE_CHECKING
from dataclasses import asdict
from ...model.huggingface import HuggingFaceDatasetPolars, build_hf_metadata, DatasetMetadata
from ...catalog.huggingface import SPEC_BY_NAME
from ...utils import get_dependencies

pl = get_dependencies(["pl"])
if TYPE_CHECKING:
    import polars as pl


DICT_2RT = [f"{i}-{j}" for i in range(1,7) for j in range(1,7) if i != j]
DICT_2RT = {place: i for i, place in enumerate(DICT_2RT)}
DICT_3RT = [f"{i}-{j}-{k}" for i in range(1,7) for j in range(1,7) for k in range(1,7) if i != j and j != k and k != i]
DICT_3RT = {place: i for i, place in enumerate(DICT_3RT)}
DATASET_NAME = "boatrace_original_20210101_20210630"


class Dataset(HuggingFaceDatasetPolars):
    metadata = build_hf_metadata(SPEC_BY_NAME[DATASET_NAME], strategy=["v3", "v1", "v2", ])

    def strategy_v1(self, df: pl.DataFrame) -> pl.DataFrame:
        self.logger.info("Create 1 rentan target")
        assert isinstance(df, pl.DataFrame)
        df = df.with_columns(pl.col("is_allboat_goal").cast(bool)).filter(pl.col("is_allboat_goal"))
        df = df.with_columns(pl.col(f"place1").cast(int))
        meta = asdict(self.metadata)
        meta["n_data"]    = df.shape[0]
        meta["n_classes"] = 6
        meta["columns_target"] = ["place1"]
        self.metadata = DatasetMetadata(**meta)
        return df
    strategy_v1.target = "polars"

    def strategy_v2(self, df: pl.DataFrame) -> pl.DataFrame:
        self.logger.info("Create 2 rentan target")
        assert isinstance(df, pl.DataFrame)
        df = df.with_columns(pl.col("is_allboat_goal").cast(bool)).filter(pl.col("is_allboat_goal"))
        df = df.with_columns(
            [pl.col(f"place{i}").cast(int).cast(str) for i in range(1,3)]
        ).with_columns(pl.concat_str([pl.col(f"place{i}") for i in range(1,3)], separator="-").alias("place12"))
        df = df.with_columns(pl.col("place12").replace(DICT_2RT).cast(pl.Int16))
        meta = asdict(self.metadata)
        meta["n_data"]    = df.shape[0]
        meta["n_classes"] = len(DICT_2RT)
        meta["columns_target"] = ["place12"]
        self.metadata = DatasetMetadata(**meta)
        return df
    strategy_v2.target = "polars"

    def strategy_v3(self, df: pl.DataFrame) -> pl.DataFrame:
        self.logger.info("Create 3 rentan target")
        assert isinstance(df, pl.DataFrame)
        df = df.with_columns(pl.col("is_allboat_goal").cast(bool)).filter(pl.col("is_allboat_goal"))
        df = df.with_columns(
            [pl.col(f"place{i}").cast(int).cast(str) for i in range(1,4)]
        ).with_columns(pl.concat_str([pl.col(f"place{i}") for i in range(1,4)], separator="-").alias("place123"))
        df = df.with_columns(pl.col("place123").replace(DICT_3RT).cast(pl.Int16))
        meta = asdict(self.metadata)
        meta["n_data"]    = df.shape[0]
        meta["n_classes"] = len(DICT_3RT)
        meta["columns_target"] = ["place123"]
        self.metadata = DatasetMetadata(**meta)
        return df
    strategy_v3.target = "polars"
