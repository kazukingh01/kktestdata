import copy
import torch
import polars as pl
from sklearn.datasets import fetch_openml
from kklogger import set_logger

from ..base import BaseDataset, DatasetMetadata
from ..utils import detect_label_mapping, apply_label_mapping
from ..catalog.openml import OpenMLSpec


LOGGER = set_logger(__name__)


class OpenMLDataset(BaseDataset):
    def _load_pandas(self):
        LOGGER.info("START")
        meta = self.metadata
        data = fetch_openml(
            name=meta.name,
            version=(meta.source_options or {}).get("version"),
            target_column=None,
            return_X_y=False,
            as_frame=True,
        )
        df = data["data"]
        assert meta.columns_feature is not None and isinstance(meta.columns_feature, (tuple, list))
        assert all(isinstance(x, str) and x in df.columns for x in meta.columns_feature)
        assert meta.columns_target  is not None and isinstance(meta.columns_target, (str, tuple, list))
        if isinstance(meta.columns_target, str):
            df = df[list(meta.columns_feature) + [meta.columns_target]]
        else:
            df = df[list(meta.columns_target) + list(meta.columns_feature)]
        # auto detect label mapping
        if len(meta.label_mapping_target) == 0:
            if "binary" in meta.supported_tasks or "multiclass" in meta.supported_tasks:
                dict_label = detect_label_mapping(df[[meta.columns_target]])
                if len(dict_label) > 0:
                    self.metadata.label_mapping_target[meta.columns_target] = copy.deepcopy(dict_label[meta.columns_target])
        if len(meta.label_mapping_feature) == 0:
            dict_label = detect_label_mapping(df[meta.columns_feature])
            for x, y in dict_label.items():
                self.metadata.label_mapping_feature[x] = copy.deepcopy(y)
        # apply label mapping
        if len(self.metadata.label_mapping_target) > 0:
            if isinstance(list(self.metadata.label_mapping_target.values())[0], dict):
                df = apply_label_mapping(df, self.metadata.label_mapping_target)
            else:
                df = apply_label_mapping(df, {meta.columns_target: self.metadata.label_mapping_target})
        if len(self.metadata.label_mapping_feature) > 0:
            df = apply_label_mapping(df, self.metadata.label_mapping_feature)
        LOGGER.info("END")
        return df

    def _load_numpy(self):
        LOGGER.info("START")
        df    = self._load_pandas()
        ndf_x = df[self.metadata.columns_feature].to_numpy()
        ndf_y = df[self.metadata.columns_target ].to_numpy()
        LOGGER.info("END")
        return ndf_x, ndf_y
    
    def _load_polars(self):
        LOGGER.info("START")
        df = self._load_pandas()
        df = pl.from_dataframe(df)
        LOGGER.info("END")
        return df

    def _load_torch(self):
        LOGGER.info("START")
        ndf_x, ndf_y = self._load_numpy()
        ndf_x = torch.from_numpy(ndf_x)
        ndf_y = torch.from_numpy(ndf_y)
        LOGGER.info("END")
        return ndf_x, ndf_y


def build_openml_metadata(spec: OpenMLSpec, strategy: str | list[str] | None=None) -> DatasetMetadata:
    return DatasetMetadata(
        name=spec.name,
        description=spec.description,
        source_type="openml",
        source_options={"version": spec.version,},
        data_type="tabular",
        supported_formats=("numpy",),
        supported_tasks=(spec.task,),
        columns_target=spec.target,
        columns_feature=spec.features,
        strategy=strategy,
        label_mapping_target={},
        label_mapping_feature={},
        revision="v1.0.0",
    )