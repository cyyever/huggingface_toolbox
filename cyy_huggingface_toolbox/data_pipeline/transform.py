import functools
import torch
from typing import Callable

import transformers
from ..model import HuggingFaceModelEvaluator
from ..tokenizer import HuggingFaceTokenizer
from cyy_torch_toolbox import (
    DatasetCollection,
    ModelType,
    TransformType,
)


def squeeze_huggingface_input(huggingface_input: dict) -> dict:
    for k, v in huggingface_input.items():
        if isinstance(v, torch.Tensor):
            huggingface_input[k] = huggingface_input[k].squeeze(dim=0)
    return huggingface_input


def apply_tokenizer_transforms(
    dc: DatasetCollection,
    model_evaluator: HuggingFaceModelEvaluator,
    max_len: int | None,
    for_input: bool,
) -> None:
    if for_input:
        batch_key = TransformType.InputBatch
        key = TransformType.Input
    else:
        batch_key = TransformType.TargetBatch
        key = TransformType.Target
    assert isinstance(model_evaluator.tokenizer, HuggingFaceTokenizer)
    assert max_len is not None
    dc.append_transform(
        functools.partial(
            model_evaluator.tokenizer.tokenizer,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        ),
        key=key,
    )
    dc.append_transform(squeeze_huggingface_input, key=key)
    collator: Callable = transformers.DataCollatorWithPadding
    match model_evaluator.model_type:
        case ModelType.TokenClassification:
            collator = transformers.DataCollatorForTokenClassification
    dc.append_transform(
        functools.partial(
            collator(
                tokenizer=model_evaluator.tokenizer.tokenizer,
                padding="max_length",
                max_length=max_len,
            )
        ),
        key=batch_key,
    )
