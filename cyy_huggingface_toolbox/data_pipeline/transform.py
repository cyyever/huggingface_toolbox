import functools
import torch
from typing import Callable, Any

import transformers
from cyy_torch_toolbox import (
    DatasetCollection,
    ModelType,
    ModelEvaluator,
    TransformType,
    default_data_extraction,
    DatasetType,
    MachineLearningPhase,
)
from cyy_torch_toolbox.data_pipeline.common import (
    int_target_to_text,
    replace_str,
)


from cyy_naive_lib.log import get_logger

from ..model import HuggingFaceModelEvaluator
from ..tokenizer import HuggingFaceTokenizer


def squeeze_huggingface_input(huggingface_input: dict) -> dict:
    for k, v in huggingface_input.items():
        if isinstance(v, torch.Tensor):
            huggingface_input[k] = huggingface_input[k].squeeze(dim=0)
    return huggingface_input


def tokenize_input(
    tokenizer: HuggingFaceTokenizer, tokenizer_call: Callable, data: Any
) -> Any:
    if "tokens" in data:
        res = {}
        res["input_ids"] = tokenizer.tokenizer.convert_tokens_to_ids(data.pop("tokens"))
        return data | res
    return tokenizer_call(data)


def collect_label(data: Any) -> Any:
    # print(data["input_ids"].shape)
    # print(data["input_ids"])
    # fdsfds
    assert len(data["ner_tags"]) == len(data["input_ids"])
    data["labels"] = data.pop("ner_tags")
    return data


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
    if not isinstance(model_evaluator.tokenizer, HuggingFaceTokenizer):
        return
    if model_evaluator.model_type != ModelType.TokenClassification:
        assert max_len is not None
        dc.append_transform(
            functools.partial(
                tokenize_input,
                model_evaluator.tokenizer,
                functools.partial(
                    model_evaluator.tokenizer.tokenizer,
                    max_length=max_len,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                ),
            ),
            key=key,
        )
        dc.append_transform(squeeze_huggingface_input, key=key)
        dc.append_transform(
            functools.partial(
                transformers.DataCollatorWithPadding(
                    tokenizer=model_evaluator.tokenizer.tokenizer,
                    padding="max_length",
                    max_length=max_len,
                )
            ),
            key=batch_key,
        )
    if model_evaluator.model_type == ModelType.TokenClassification:
        dc.append_transform(
            functools.partial(
                tokenize_input,
                model_evaluator.tokenizer,
                model_evaluator.tokenizer.tokenizer,
            ),
            key=key,
        )
        dc.append_transform(collect_label, key=key)
        dc.append_transform(
            functools.partial(
                transformers.DataCollatorForTokenClassification(
                    tokenizer=model_evaluator.tokenizer.tokenizer,
                )
            ),
            key=batch_key,
        )


def huggingface_data_extraction(model_type: ModelType, data: Any) -> dict:
    if model_type in (ModelType.TokenClassification,):
        match data:
            case {"data": data, "index": index}:
                data.pop("id", None)
                return {"index": index, "input": data}
        raise RuntimeError(data)
    return default_data_extraction(data)


def add_text_extraction(dc: DatasetCollection, model_evaluator: Any) -> None:
    if model_evaluator is not None:
        assert isinstance(model_evaluator, ModelEvaluator)
        for _, transform in dc.foreach_transform():
            transform.clear(TransformType.ExtractData)
            transform.append(
                key=TransformType.ExtractData,
                transform=functools.partial(
                    huggingface_data_extraction, model_evaluator.model_type
                ),
            )

    assert dc.dataset_type == DatasetType.Text
    dataset_name: str = dc.name.lower()
    # InputText
    if dataset_name == "imdb":
        dc.append_transform(
            functools.partial(replace_str, old="<br />", new=""),
            key=TransformType.InputText,
        )


def get_label_to_text_mapping(dataset_name: str) -> dict | None:
    match dataset_name.lower():
        case "multi_nli":
            return {0: "entailment", 1: "neutral", 2: "contradiction"}
        case "imdb":
            return {0: "negative", 1: "positive"}
    return None


def add_text_transforms(
    dc: DatasetCollection, model_evaluator: HuggingFaceModelEvaluator
) -> None:
    assert dc.dataset_type in (DatasetType.Text, DatasetType.CodeText)
    dataset_name: str = dc.name.lower()
    # InputText
    assert model_evaluator.model_type is not None

    # Input && InputBatch
    input_max_len = dc.dataset_kwargs.get("input_max_len", None)
    if input_max_len is not None:
        get_logger().info("use input text max_len %s", input_max_len)
    apply_tokenizer_transforms(
        dc=dc, model_evaluator=model_evaluator, max_len=input_max_len, for_input=True
    )

    # Target
    if model_evaluator.model_type == ModelType.TextGeneration:
        mapping = get_label_to_text_mapping(dataset_name)
        if mapping is not None:
            dc.append_transform(
                functools.partial(int_target_to_text, mapping=mapping),
                key=TransformType.Target,
            )
        elif isinstance(
            dc.get_dataset_util(phase=MachineLearningPhase.Training).get_sample_label(
                0
            ),
            int,
        ):
            dc.append_transform(int_target_to_text, key=TransformType.Target)
        max_len = dc.dataset_kwargs.get("output_max_len", None)
        get_logger().info("use output text max len %s", max_len)
        apply_tokenizer_transforms(
            dc=dc, model_evaluator=model_evaluator, max_len=max_len, for_input=False
        )
