from kktestdata import DatasetRegistry


if __name__ == "__main__":
    reg = DatasetRegistry()
    for name in reg._datasets.keys():
        dataset = reg.create(name=name)
        x, y = dataset.load_data(format="numpy")
        print(dataset.to_display())