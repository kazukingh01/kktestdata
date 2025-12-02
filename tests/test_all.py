from kktestdata import DatasetRegistry
from kktestdata.catalog.openml import OPENML_SPECS


if __name__ == "__main__":
    reg = DatasetRegistry()
    for name in [spec.name for spec in OPENML_SPECS]:
        dataset = reg.create(name=name)
        x, y = dataset.load_data(format="numpy")
        print(dataset.to_display())