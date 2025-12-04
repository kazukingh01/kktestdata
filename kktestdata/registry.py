import importlib.util
import sys
from inspect import isclass
from pathlib import Path
from types import ModuleType
from typing import Iterable, TYPE_CHECKING
from kklogger import set_logger
from .base import BaseDataset, DatasetError, DatasetMetadata, to_display, to_dict
from .utils import get_dependencies
from .check import check_random_seed
from .helpers import RANDOM_SEED

# import dependencies if it's ready to use
pd = get_dependencies(["pd"])
if TYPE_CHECKING:
    import pandas as pd


LOGGER = set_logger(__name__)


class DatasetNotFoundError(DatasetError):
    def __init__(self, name: str):
        super().__init__(f"Dataset '{name}' not found")
        self.name = name


class DatasetRegistry:
    def __init__(self, datasets_dir: str | Path | None = None, auto_discover: bool = True):
        self.datasets_dir = (
            Path(datasets_dir).resolve()
            if datasets_dir is not None
            else Path(__file__).resolve().parent / "dataset"
        )
        self._datasets: dict[str, type[BaseDataset]] = {}
        self._load_errors: dict[str, Exception] = {}
        if auto_discover:
            self.discover()

    def discover(self, reload: bool = False) -> None:
        assert isinstance(reload, bool)
        if reload:
            self._datasets.clear()
            self._load_errors.clear()

        if not self.datasets_dir.exists():
            raise FileNotFoundError(f"datasets directory not found: {self.datasets_dir}")

        for path in self._iter_dataset_files():
            try:
                module = self._load_module(path)
                dataset_cls = self._extract_dataset_class(module)
                self.register(dataset_cls)
            except Exception as exc:  # noqa: BLE001
                self._load_errors[path.stem] = exc
                LOGGER.warning("Skipping dataset %s: %s", path.name, exc)

    def register(self, dataset_cls: type[BaseDataset]) -> None:
        assert isinstance(dataset_cls, type) and issubclass(dataset_cls, BaseDataset)
        meta = getattr(dataset_cls, "metadata")
        assert isinstance(meta, DatasetMetadata)
        if meta.name in self._datasets:
            raise DatasetError(f"Dataset with name '{meta.name}' is already registered")
        self._datasets[meta.name] = dataset_cls

    def get_class(self, name: str) -> type[BaseDataset]:
        assert isinstance(name, str) and name
        if name not in self._datasets:
            raise DatasetNotFoundError(name)
        return self._datasets[name]

    def create(self, name: str, seed: int = RANDOM_SEED) -> BaseDataset:
        assert isinstance(name, str) and name
        check_random_seed(seed)
        LOGGER.info(f"Creating dataset {name}", color=["GREEN"])
        dataset_cls = self.get_class(name)
        return dataset_cls(dataset_cls.metadata, seed=seed)

    def names(self) -> tuple[str, ...]:
        return tuple(self._datasets.keys())

    def info(self, name: str | None = None) -> list[dict] | dict:
        assert name is None or isinstance(name, str)
        if name is not None:
            return to_display(self.get_class(name).metadata)
        list_dict = [to_dict(self.get_class(name).metadata, list_keys = [
            "name", "source_type", "data_type", "supported_formats", "supported_task", 
            "n_data", "n_classes", "n_target", "n_features", "n_null_columns"
        ]) for name in self.names()]
        return pd.DataFrame(list_dict).to_string(index=False)

    def _iter_dataset_files(self) -> Iterable[Path]:
        for path in sorted(self.datasets_dir.rglob("*.py")):
            if path.name == "__init__.py" or path.name.startswith("_"):
                continue
            if "__pycache__" in path.parts or "utils" in path.parts:
                continue
            yield path

    def _load_module(self, path: Path) -> ModuleType:
        relative = path.relative_to(self.datasets_dir).with_suffix("")
        module_name = ".".join((__package__, "dataset", *relative.parts))
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise DatasetError(f"Could not create module spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _extract_dataset_class(self, module: ModuleType) -> type[BaseDataset]:
        candidates = [
            obj
            for obj in module.__dict__.values()
            if isclass(obj)
            and issubclass(obj, BaseDataset)
            and obj is not BaseDataset
            and obj.__module__ == module.__name__
        ]
        if not candidates:
            raise DatasetError(f"No BaseDataset subclass found in module {module.__name__}")
        if len(candidates) > 1:
            raise DatasetError(f"Multiple BaseDataset subclasses found in module {module.__name__}")
        dataset_cls = candidates[0]
        meta = getattr(dataset_cls, "metadata", None)
        if not isinstance(meta, DatasetMetadata):
            raise DatasetError(f"{dataset_cls.__name__} must define `metadata: DatasetMetadata`")
        return dataset_cls

    def _iter_metadata(self) -> Iterable[DatasetMetadata]:
        return (cls.metadata for cls in self._datasets.values())


__all__ = ["DatasetRegistry", "DatasetNotFoundError"]
