from ...model.openml import OpenMLDataset
from ...catalog.openml import SPEC_BY_NAME
from ...model.openml import build_openml_metadata


DATASET_NAME = "tamilnadu-electricity"


class Dataset(OpenMLDataset):
    metadata = build_openml_metadata(SPEC_BY_NAME[DATASET_NAME])
