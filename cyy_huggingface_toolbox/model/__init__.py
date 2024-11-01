from typing import Any

import transformers
from cyy_torch_toolbox import ModelEvaluator

from .huggingface_evaluator import HuggingFaceModelEvaluator
from .huggingface_model import get_huggingface_constructor


def get_model_evaluator(model, **kwargs: Any) -> ModelEvaluator | None:
    if isinstance(model, transformers.PreTrainedModel):
        return HuggingFaceModelEvaluator(model=model, **kwargs)
    return None


__all__ = [
    "HuggingFaceModelEvaluator",
    "get_huggingface_constructor",
    "get_model_evaluator",
]
