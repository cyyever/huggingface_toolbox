import functools
from typing import Any

import torch
import transformers
from cyy_naive_lib.log import log_info
from cyy_torch_toolbox import (
    DatasetCollection,
    DatasetType,
    MachineLearningPhase,
    ModelEvaluator,
    ModelType,
    TransformType,
    default_data_extraction,
)
from cyy_torch_toolbox.data_pipeline.common import (
    int_target_to_text,
    replace_str,
)

from ..model import HuggingFaceModelEvaluator
from ..tokenizer import HuggingFaceTokenizer


def squeeze_huggingface_input(huggingface_input: dict) -> dict:
    for k, v in huggingface_input.items():
        if isinstance(v, torch.Tensor):
            huggingface_input[k] = huggingface_input[k].squeeze(dim=0)
    return huggingface_input


def tokenize_and_align_labels(
    tokenizer: transformers.PreTrainedTokenizerFast, examples
):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    word_ids = tokenized_inputs.word_ids(
        batch_index=0
    )  # Map tokens to their respective word.
    previous_word_idx: None | int = None
    label = examples["ner_tags"]
    label_ids: list[int] = []
    for word_idx in word_ids:  # Set the special tokens to -100.
        if word_idx is None:
            label_ids.append(-100)
        elif (
            word_idx != previous_word_idx
        ):  # Only label the first token of a given word.
            label_ids.append(label[word_idx])
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs


# def collect_label(data: Any) -> Any:
#     assert len(data["ner_tags"]) == len(data["input_ids"])
#     data["labels"] = data.pop("ner_tags")
#     return data


def apply_tokenizer_transforms(
    dc: DatasetCollection,
    model_evaluator: HuggingFaceModelEvaluator,
    max_len: int | None,
    for_input: bool,
) -> None:
    if not isinstance(model_evaluator.tokenizer, HuggingFaceTokenizer):
        return
    if for_input:
        batch_key = TransformType.InputBatch
        key = TransformType.Input
    else:
        batch_key = TransformType.TargetBatch
        key = TransformType.Target
    assert max_len is not None
    if model_evaluator.model_type == ModelType.TokenClassification:
        dc.append_transform(
            functools.partial(
                tokenize_and_align_labels, model_evaluator.tokenizer.tokenizer
            ),
            key=key,
        )
        dc.append_transform(
            functools.partial(
                transformers.DataCollatorForLanguageModeling(
                    tokenizer=model_evaluator.tokenizer.tokenizer,
                    padding="max_length",
                    max_length=max_len,
                )
            ),
            key=batch_key,
        )
        return
    dc.append_transform(
        functools.partial(
            model_evaluator.tokenizer.tokenizer,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        ),
        key=batch_key,
    )
    if model_evaluator.model_type == ModelType.CausalLM:
        dc.append_transform(
            functools.partial(
                transformers.DatasetCollectionF(
                    tokenizer=model_evaluator.tokenizer.tokenizer,
                    padding="max_length",
                    max_length=max_len,
                )
            ),
            key=batch_key,
        )
        return
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
        log_info("use input text max_len %s", input_max_len)
    apply_tokenizer_transforms(
        dc=dc, model_evaluator=model_evaluator, max_len=input_max_len, for_input=True
    )

    # # Target
    # if model_evaluator.model_type == ModelType.TextGeneration:
    #     mapping = get_label_to_text_mapping(dataset_name)
    #     if mapping is not None:
    #         dc.append_transform(
    #             functools.partial(int_target_to_text, mapping=mapping),
    #             key=TransformType.Target,
    #         )
    #     elif isinstance(
    #         dc.get_dataset_util(phase=MachineLearningPhase.Training).get_sample_label(
    #             0
    #         ),
    #         int,
    #     ):
    #         dc.append_transform(int_target_to_text, key=TransformType.Target)
    #     max_len = dc.dataset_kwargs.get("output_max_len", None)
    #     log_info("use output text max len %s", max_len)
    #     apply_tokenizer_transforms(
    #         dc=dc, model_evaluator=model_evaluator, max_len=max_len, for_input=False
    #     )
