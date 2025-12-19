from ...model.huggingface import build_hf_metadata
from ...catalog.huggingface import SPEC_BY_NAME
from .boatrace_original_20210101_20210630 import Dataset as Dataset1


DATASET_NAME = "boatrace_original_20210701_20211230"


class Dataset(Dataset1):
    metadata = build_hf_metadata(SPEC_BY_NAME[DATASET_NAME], strategy=["v3", "v1", "v2", ])
