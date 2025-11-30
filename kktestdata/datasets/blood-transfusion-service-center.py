import numpy as np
from sklearn.datasets import fetch_openml

from ..base import BaseDataset, DatasetMetadata


DATASET_NAME = "blood-transfusion-service-center"


class Dataset(BaseDataset):
    metadata = DatasetMetadata(
        name=DATASET_NAME,
        description="Blood donation prediction dataset (OpenML 1464).",
        source_type="openml",
        source_options=None,
        data_type="tabular",
        supported_formats=("numpy",),
        supported_tasks=("binary",),
        columns_target="Class",
        columns_feature=("Recency", "Frequency", "Monetary", "Time"),
        strategy=None,
        label_mapping={"1": 0, "2": 1},
        revision="v1.0.0",
    )

    def _load_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        source_name = self.metadata.source_options.get("name", self.metadata.name)
        source_version = self.metadata.source_options.get("version")
        x, y = fetch_openml(
            name=source_name,
            version=source_version,
            return_X_y=True,
            as_frame=False,
        )
        return x, np.vectorize(lambda x: self.metadata.label_mapping.get(x))(y).astype(int)
