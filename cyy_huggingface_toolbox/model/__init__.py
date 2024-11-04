from typing import Any

import transformers


import functools
from collections.abc import Callable

from cyy_naive_lib.log import log_debug
from cyy_torch_toolbox import DatasetCollection, DatasetType, Factory
from cyy_torch_toolbox.model import (
    create_model,
    global_model_evaluator_factory,
    global_model_factory,
    ModelEvaluator,
)
from cyy_torch_toolbox.model.repositary import get_model_info
from .huggingface_evaluator import HuggingFaceModelEvaluator
from .huggingface_model import get_huggingface_constructor

from ..tokenizer import HuggingFaceTokenizer


def __get_model_evaluator(
    parent_factory: Callable | None, model, **kwargs: Any
) -> HuggingFaceModelEvaluator | ModelEvaluator | None:
    if isinstance(model, transformers.PreTrainedModel):
        return HuggingFaceModelEvaluator(model=model, **kwargs)
    if parent_factory is not None:
        return parent_factory(model=model, **kwargs)
    return None


global_model_evaluator_factory.register(
    DatasetType.Text,
    functools.partial(
        __get_model_evaluator, global_model_evaluator_factory.get(DatasetType.Text)
    ),
)


model_constructors = get_model_info().get(DatasetType.Text, {})


class HuggingFaceModelFactory(Factory):
    def __init__(self, parent_factory: None | Factory = None) -> None:
        super().__init__()
        self.__parent_factory = parent_factory

    def get(self, key: str, case_sensitive: bool = False) -> Callable | None:
        if self.__parent_factory is not None:
            res = self.__parent_factory.get(key=key, case_sensitive=case_sensitive)
            if res is not None:
                return res
        assert not case_sensitive
        model_name = key
        res = get_huggingface_constructor(model_name)
        if res is not None:
            constructor, name = res
            print("name is", name)
            return functools.partial(
                self.__create_text_model, name=name, constructor=constructor
            )
        return None

    def __create_text_model(
        self,
        name,
        constructor,
        dataset_collection: DatasetCollection,
        **kwargs: Any,
    ) -> dict:
        final_model_kwargs: dict = kwargs
        tokenizer_kwargs = dataset_collection.dataset_kwargs.get("tokenizer", {})
        tokenizer_kwargs["name"] = name
        tokenizer = HuggingFaceTokenizer(tokenizer_kwargs)
        log_debug("tokenizer is %s", tokenizer)

        if tokenizer is not None and hasattr(tokenizer, "itos"):
            token_num = len(tokenizer.get_vocab())
            for k in ("num_embeddings", "token_num"):
                if k not in kwargs:
                    final_model_kwargs[k] = token_num
        model = create_model(constructor, **final_model_kwargs)

        return {"model": model, "tokenizer": tokenizer}


for dataset_type in (DatasetType.Text, DatasetType.CodeText):
    global_model_factory[dataset_type] = HuggingFaceModelFactory(
        parent_factory=global_model_factory.get(dataset_type, None)
    )


__all__ = [
    "HuggingFaceModelEvaluator",
]
