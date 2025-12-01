from ..base import BaseDataset
from ._openml_catalog import SPEC_BY_NAME
from ._openml_common import build_openml_metadata, load_openml_numpy


DATASET_NAME = "blood-transfusion-service-center"


class Dataset(BaseDataset):
    metadata = build_openml_metadata(SPEC_BY_NAME[DATASET_NAME])

    def _load_numpy(self):
        return load_openml_numpy(self)
