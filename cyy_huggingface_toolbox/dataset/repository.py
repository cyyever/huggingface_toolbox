import copy
import functools
import os
from typing import Any

import dill
from cyy_naive_lib.log import log_info
from cyy_torch_toolbox.dataset import DatasetFactory
from datasets import Split, load_dataset_builder
from datasets import load_dataset as load_hugging_face_dataset


class HunggingFaceFactory(DatasetFactory):
    def get(
        self, key: Any, case_sensitive: bool = True, default: Any = None, **kwargs: Any
    ) -> Any:
        assert case_sensitive
        assert default is None
        cache_dir = kwargs.pop("cache_dir", None)
        assert cache_dir is not None

        if not self.__has_dataset(key, cache_dir, kwargs["dataset_kwargs"]):
            return None

        return functools.partial(self.__get_dataset, path=key, cache_dir=cache_dir)

    @classmethod
    def __get_dataset(
        cls, path: str, cache_dir: str, split: Any, name: Any | None = None, **kwargs
    ) -> Any:
        if "train" in split:
            split = Split.TRAIN
        elif "val" in split:
            split = Split.VALIDATION
        elif "test" in split:
            split = Split.TEST
        kwargs["split"] = split
        kwargs["name"] = name
        kwargs["cache_dir"] = cache_dir
        file_key = f"{str(split).lower()}_files"
        if kwargs.get(file_key):
            kwargs = copy.deepcopy(kwargs)
            data_files = kwargs.pop(file_key)
            assert data_files
            if isinstance(data_files, str):
                data_files = [data_files]
            path = os.path.splitext(data_files[0])[1][1:]
            for k in list(kwargs.keys()):
                if "_files" in k:
                    kwargs.pop(k)
            kwargs["split"] = Split.TRAIN
            kwargs["data_files"] = data_files
            kwargs["cache_dir"] = None
        try:
            dataset = load_hugging_face_dataset(
                path=path,
                **kwargs,
            )
            if "data_files" in kwargs:
                setattr(dataset, file_key, kwargs["data_files"])
        except BaseException as e:
            if cls.__has_dataset(key=path, cache_dir=cache_dir, dataset_kwargs=kwargs):
                if file_key not in kwargs:
                    return None
            log_info("exception is %s", e)
            raise e
        if not os.path.isfile(cls.__dataset_cache_file(cache_dir, split)):
            os.makedirs(cls.__dataset_cache_dir(cache_dir), exist_ok=True)
            with open(cls.__dataset_cache_file(cache_dir, split), "wb") as f:
                dill.dump(True, f)
        return dataset

    @classmethod
    def __has_dataset(cls, key: Any, cache_dir: str, dataset_kwargs: dict) -> bool:
        if key.startswith("hugging_face_"):
            return True
        if os.path.exists(cls.__dataset_cache_dir(cache_dir)):
            return True
        try:
            load_dataset_builder(path=key, name=dataset_kwargs.get("name"))
            return True
        except BaseException as e:
            log_info("exception is %s", e)
        return False

    @classmethod
    def __dataset_cache_dir(cls, cache_dir: str) -> str:
        return os.path.join(cache_dir, ".cache", "hg_cache")

    @classmethod
    def __dataset_cache_file(cls, cache_dir: str, split: Any) -> str:
        return os.path.join(cls.__dataset_cache_dir(cache_dir), str(split))
