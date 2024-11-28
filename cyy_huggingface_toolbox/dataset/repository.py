import functools
import os
from typing import Any

import dill
from cyy_torch_toolbox.dataset import DatasetFactory
from datasets import Split, load_dataset_builder
from datasets import load_dataset as load_hugging_face_dataset


class HunggingFaceFactory(DatasetFactory):
    def get(
        self, key: str, case_sensitive: bool = True, cache_dir: str | None = None
    ) -> Any:
        assert case_sensitive
        assert cache_dir is not None
        if not self.__has_dataset(key, cache_dir):
            return None

        return functools.partial(self.__get_dataset, path=key, cache_dir=cache_dir)

    @classmethod
    def __get_dataset(cls, path: str, cache_dir: str, split: Any, **kwargs) -> Any:
        if "train" in split:
            split = Split.TRAIN
        elif "val" in split:
            split = Split.VALIDATION
        elif "test" in split:
            split = Split.TEST
        dataset = load_hugging_face_dataset(
            path=path, split=split, cache_dir=cache_dir, **kwargs
        )
        if not os.path.isfile(cls.__dataset_cache_file(cache_dir, split)):
            os.makedirs(
                os.path.join(cls.__dataset_cache_dir(cache_dir), ".cache", "hg_cache"),
                exist_ok=True,
            )
            with open(cls.__dataset_cache_file(cache_dir, split), "wb") as f:
                dill.dump(True, f)
        return dataset

    @classmethod
    def __has_dataset(cls, key: Any, cache_dir: str) -> bool:
        if key.endswith(".json"):
            return True
        if os.path.exists(cls.__dataset_cache_dir(cache_dir)):
            return True
        try:
            load_dataset_builder(path=key)
            return True
        except BaseException:
            pass
        return False

    def get_similar_keys(self, key: str) -> list[str]:
        return []

    @classmethod
    def __dataset_cache_dir(cls, cache_dir: str) -> str:
        return os.path.join(cache_dir, ".cache", "hg_cache")

    @classmethod
    def __dataset_cache_file(cls, cache_dir: str, split: Any) -> str:
        return os.path.join(cls.__dataset_cache_dir(cache_dir), str(split))
