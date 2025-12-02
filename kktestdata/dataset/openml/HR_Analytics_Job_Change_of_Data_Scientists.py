from ...model.openml import OpenMLDataset
from ...catalog.openml import SPEC_BY_NAME
from ...model.openml import build_openml_metadata


DATASET_NAME = "HR_Analytics_Job_Change_of_Data_Scientists"


class Dataset(OpenMLDataset):
    metadata = build_openml_metadata(SPEC_BY_NAME[DATASET_NAME])
