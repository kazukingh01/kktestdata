from ...model.openml import OpenMLDataset
from ...catalog.openml import SPEC_BY_NAME
from ...model.openml import build_openml_metadata


DATASET_NAME = "customer_satisfaction_in_airline"


class Dataset(OpenMLDataset):
    metadata = build_openml_metadata(SPEC_BY_NAME[DATASET_NAME])
