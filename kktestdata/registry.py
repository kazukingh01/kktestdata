import importlib.util
import sys
from dataclasses import asdict
from inspect import isclass
from pathlib import Path
from types import ModuleType
from typing import Iterable
from kklogger import set_logger

from .base import BaseDataset, DatasetError, DatasetMetadata, DatasetNotFoundError


LOGGER = set_logger(__name__)


class DatasetRegistry:
    """
    Discovers dataset implementations under ``datasets/*.py`` and exposes them
    through a simple registry interface.
    """

    def __init__(self, datasets_dir: str | Path | None = None, auto_discover: bool = True):
        self.datasets_dir = (
            Path(datasets_dir).resolve()
            if datasets_dir is not None
            else Path(__file__).resolve().parent / "datasets"
        )
        self._datasets: dict[str, type[BaseDataset]] = {}
        self._load_errors: dict[str, Exception] = {}

        if auto_discover:
            self.discover()

    def discover(self, reload: bool = False) -> None:
        """
        Import every ``datasets/*.py`` file and register a single BaseDataset subclass
        defined in each module. Modules that fail to import are skipped and recorded
        in ``load_errors``.
        """
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
        """
        Register a dataset class explicitly. The class must expose a ``metadata``
        attribute of type DatasetMetadata.
        """
        if not isclass(dataset_cls) or not issubclass(dataset_cls, BaseDataset):
            raise DatasetError(f"{dataset_cls!r} is not a BaseDataset subclass")

        meta = getattr(dataset_cls, "metadata", None)
        if not isinstance(meta, DatasetMetadata):
            raise DatasetError(f"{dataset_cls.__name__} must define `metadata: DatasetMetadata`")

        if meta.name in self._datasets:
            raise DatasetError(f"Dataset with name '{meta.name}' is already registered")

        self._datasets[meta.name] = dataset_cls

    def get_class(self, name: str) -> type[BaseDataset]:
        """Return the registered dataset class for ``name``."""
        try:
            return self._datasets[name]
        except KeyError as exc:
            raise DatasetNotFoundError(name) from exc

    def create(self, name: str) -> BaseDataset:
        """
        Instantiate a dataset by name. The dataset class is initialized with its
        declared metadata.
        """
        LOGGER.info(f"Creating dataset {name}", color=["GREEN"])
        dataset_cls = self.get_class(name)
        return dataset_cls(dataset_cls.metadata)

    def get_metadata(self, name: str) -> DatasetMetadata:
        """Return the metadata for a registered dataset."""
        dataset_cls = self.get_class(name)
        return dataset_cls.metadata

    def names(self) -> tuple[str, ...]:
        """Return registered dataset names."""
        return tuple(self._datasets.keys())

    def info(self, name: str | None = None) -> list[dict] | dict:
        """
        Return metadata information as dictionaries. When ``name`` is omitted,
        all registered datasets are included; otherwise only the selected one.
        """
        if name is not None:
            return self._metadata_to_dict(self.get_metadata(name))
        return [self._metadata_to_dict(meta) for meta in self._iter_metadata()]

    @property
    def load_errors(self) -> dict[str, Exception]:
        """Datasets that failed to load during discovery."""
        return dict(self._load_errors)

    def _iter_dataset_files(self) -> Iterable[Path]:
        return (
            path
            for path in sorted(self.datasets_dir.glob("*.py"))
            if path.name not in {"__init__.py", "__pycache__"} and not path.name.startswith("_")
        )

    def _load_module(self, path: Path) -> ModuleType:
        module_name = f"{__package__}.datasets.{path.stem}"
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
            if isclass(obj) and issubclass(obj, BaseDataset) and obj is not BaseDataset
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

    def _metadata_to_dict(self, meta: DatasetMetadata) -> dict:
        return asdict(meta)


__all__ = ["DatasetRegistry"]
