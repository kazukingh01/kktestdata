from sklearn.datasets import fetch_openml
from kktestdata.catalog.openml import OPENML_SPECS
for spec in OPENML_SPECS:
    data = fetch_openml(
        name=spec.name,
        version=spec.version,
        return_X_y=False,
        as_frame=True,
    )
    print(spec.name)
    print(data.feature_names)
    print(data.target_names)
    print(data.target.shape)
    print("--------------------------------")
