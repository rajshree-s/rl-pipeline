from abc import ABC, abstractmethod
from typing import cast

from rl_pipeline.utils import dataset_cache_dir

from datasets import load_dataset, Dataset as HFDataset


class Dataset(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def splits(self) -> tuple[str, ...]: ...

    # TODO: still figuring where this should be
    @property
    @abstractmethod
    def prompt(self) -> str: ...

    @abstractmethod
    def _transform_from_raw(self, dataset: HFDataset) -> HFDataset:
        """
        Should return flattened dataset with question, answer pairs
        """
        ...

    @classmethod
    def download(cls):
        self = cls.__new__(cls)
        print(f"Downlaoding dataset '{self.name}'")
        load_dataset(self.name, cache_dir=dataset_cache_dir().as_uri())


    def load_dataset(self, split: str, no_of_records: int | None = None) -> HFDataset:
        assert self.has_split(split)
        raw_dataset = cast(
            HFDataset,
            load_dataset(
                self.name, cache_dir=dataset_cache_dir().as_uri(), split=split
            ),
        )
        raw_dataset = raw_dataset.train_test_split(no_of_records)["test"] if no_of_records else raw_dataset

        return self._transform_from_raw(raw_dataset)

    def has_split(self, split: str) -> bool:
        return split in self.splits
