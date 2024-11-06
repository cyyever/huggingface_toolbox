from typing import Any
import functools
from collections.abc import Callable
import transformers

from cyy_naive_lib.log import log_debug
from cyy_torch_toolbox import DatasetCollection, DatasetType, Factory, ModelType
from cyy_torch_toolbox.model import (
    create_model,
    global_model_evaluator_factory,
    global_model_factory,
    ModelEvaluator,
)
from .huggingface_evaluator import HuggingFaceModelEvaluator
from .huggingface_model import get_huggingface_constructor
from ..tokenizer import HuggingFaceTokenizer


def __get_model_evaluator(
    model, **kwargs: Any
) -> HuggingFaceModelEvaluator | ModelEvaluator | None:
    if isinstance(model, transformers.PreTrainedModel):
        return HuggingFaceModelEvaluator(model=model, **kwargs)
    return None


for dataset_type in (DatasetType.Text, DatasetType.CodeText):
    global_model_evaluator_factory.register(
        dataset_type,
        [__get_model_evaluator],
    )


class HuggingFaceModelFactory(Factory):
    def get(self, key: str, case_sensitive: bool = True) -> dict | None:
        assert case_sensitive
        model_name = key
        res = get_huggingface_constructor(model_name)
        if res is None:
            return None
        constructor, name, model_type = res
        return {
            "constructor": functools.partial(
                self.__create_model,
                real_name=name,
                model_type=model_type,
                constructor=constructor,
            ),
            "model_type": model_type,
        }

    def __create_model(
        self,
        real_name: str,
        model_type: ModelType,
        constructor: Callable,
        dataset_collection: DatasetCollection,
        **kwargs: Any,
    ) -> dict:
        final_model_kwargs: dict = kwargs
        tokenizer_kwargs = dataset_collection.dataset_kwargs.get("tokenizer", {})
        tokenizer_kwargs["name"] = real_name
        tokenizer = HuggingFaceTokenizer(tokenizer_kwargs)
        log_debug("tokenizer is %s", tokenizer)

        if tokenizer is not None and hasattr(tokenizer, "itos"):
            token_num = len(tokenizer.get_vocab())
            for k in ("num_embeddings", "token_num"):
                if k not in kwargs:
                    final_model_kwargs[k] = token_num
        model = create_model(constructor, **final_model_kwargs)

        return {"model": model, "tokenizer": tokenizer, "model_type": model_type}


for dataset_type in (DatasetType.Text, DatasetType.CodeText):
    if dataset_type not in global_model_factory:
        global_model_factory[dataset_type] = []
    global_model_factory[dataset_type].append(HuggingFaceModelFactory())
