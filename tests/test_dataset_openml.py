import argparse
from sklearn.datasets import fetch_openml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=lambda x: x.split(','), required=True)
    parser.add_argument("-v", "--version", type=lambda x: [int(y) for y in x.split(',')], required=False)
    args = parser.parse_args()
    if args.version is not None:
        assert len(args.version) == len(args.name)
    else:
        args.version = [None] * len(args.name)
    for name, version in zip(args.name, args.version):
        if version is not None:
            data = fetch_openml(
                name=name,
                version=version,
                return_X_y=False,
                as_frame=True,
            )
        else:
            data = fetch_openml(
                name=name,
                return_X_y=False,
                as_frame=True,
            )
        print(name)
        print(data.feature_names)
        print(data.target_names)
        print(data.target)
        print("--------------------------------")
