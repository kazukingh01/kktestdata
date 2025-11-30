import numpy as np
from sklearn.datasets import fetch_openml
from ..base import BaseDataset, DatasetMetadata


class Dataset(BaseDataset):
    metadata = DatasetMetadata(
        name=__name__,
        description=__name__,
        source_type="openml",
        source_options=None,
        data_type="tabular",
        supported_formats=("numpy"),
        supported_tasks=("binary"),
        columns_target="target",
        columns_feature=("v1", "v2", "v3", "v4"),
        strategy=None,
        label_mapping={"1": 0, "2": 1},
        revision="v1.0.0",
    )
    def _load_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        x, y = fetch_openml(
            name=self.metadata.name,
            return_X_y=True,
            as_frame=False,
        )
        return x, np.vectorize(lambda x: self.metadata.label_mapping.get(x))(y).astype(int)
