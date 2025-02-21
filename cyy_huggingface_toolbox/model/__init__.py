import copy
import functools
from collections.abc import Callable
from typing import Any

import transformers
from cyy_naive_lib.log import log_debug, log_info, log_warning
from cyy_torch_toolbox import DatasetCollection, DatasetType, Factory, ModelType
from cyy_torch_toolbox.model import (
    ModelEvaluator,
    create_model,
    global_model_evaluator_factory,
    global_model_factory,
)

from ..tokenizer import HuggingFaceTokenizer
from .evaluator import HuggingFaceModelEvaluator
from .finetune_evaluator import HuggingFaceModelEvaluatorForFinetune
from .model import get_huggingface_constructor

__all__ = ["HuggingFaceModelEvaluator", "HuggingFaceModelEvaluatorForFinetune"]


def __get_model_evaluator(
    model: Any, **kwargs: Any
) -> HuggingFaceModelEvaluator | ModelEvaluator | None:
    if isinstance(model, transformers.PreTrainedModel):
        if "finetune_modules" in kwargs:
            return HuggingFaceModelEvaluatorForFinetune(model=model, **kwargs)
        log_warning("Not finetune")
        return HuggingFaceModelEvaluator(model=model, **kwargs)
    return None


for dataset_type in (DatasetType.Text, DatasetType.CodeText):
    global_model_evaluator_factory.register(
        dataset_type,
        [__get_model_evaluator],
    )


class HuggingFaceModelFactory(Factory):
    def get(
        self, key: str, case_sensitive: bool = True, default: Any = None
    ) -> dict | None:
        assert case_sensitive
        assert default is None
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
        dataset_collection: DatasetCollection | None = None,
        **kwargs: Any,
    ) -> dict:
        final_model_kwargs: dict = copy.deepcopy(kwargs)
        final_model_kwargs.pop("name", None)
        tokenizer_kwargs = {}
        if dataset_collection is not None:
            tokenizer_kwargs = dataset_collection.dataset_kwargs.get("tokenizer", {})
        tokenizer_kwargs["name"] = real_name
        tokenizer = HuggingFaceTokenizer(tokenizer_kwargs)
        log_info("tokenizer is %s", type(tokenizer.tokenizer))

        model = create_model(constructor, **final_model_kwargs)

        return {"model": model, "tokenizer": tokenizer, "model_type": model_type}


for dataset_type in (DatasetType.Text, DatasetType.CodeText):
    if dataset_type not in global_model_factory:
        global_model_factory[dataset_type] = []
    global_model_factory[dataset_type].append(HuggingFaceModelFactory())
