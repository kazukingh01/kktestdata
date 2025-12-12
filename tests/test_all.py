from kktestdata import DatasetRegistry


if __name__ == "__main__":
    reg = DatasetRegistry()
    for name in reg._datasets.keys():
        dataset = reg.create(name=name)
        x1, y1, x2, y2, x3, y3 = dataset.load_data(format="numpy", split_type="valid", test_size=0.3, valid_size=0.3)
        print(dataset.to_display())