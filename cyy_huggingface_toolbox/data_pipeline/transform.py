import functools
import os
from typing import Any

import torch
import transformers
from cyy_naive_lib.log import log_debug, log_info
from cyy_torch_toolbox import (
    BatchTransform,
    DatasetCollection,
    MachineLearningPhase,
    ModelType,
    TextDatasetCollection,
    Transform,
)
from cyy_torch_toolbox.data_pipeline.common import (
    replace_str,
)

from ..model import HuggingFaceModelEvaluator


def merge_batch_encodings(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    assert len(data) == 1 and "input" in data
    encodings = data["input"]
    assert all(isinstance(b, transformers.BatchEncoding) for b in encodings)

    result = {}
    for k in encodings[0]:
        result[k] = torch.concat([b[k] for b in encodings], dim=0)
    result["labels"] = result["labels"].flatten()
    log_debug("input_ids shape %s", result["input_ids"].shape)
    log_debug("labels shape %s", result["labels"].shape)

    return transformers.BatchEncoding(result)


def tokenize_and_align_labels(
    tokenizer: transformers.PreTrainedTokenizerFast,
    tokenizer_kwargs: Any,
    labels: dict,
    phase: MachineLearningPhase,
    background_label: str,
    example,
) -> transformers.BatchEncoding:
    tokenized_inputs = tokenizer(
        example["tokens"], is_split_into_words=True, **tokenizer_kwargs
    )

    word_ids = tokenized_inputs.word_ids(
        batch_index=0
    )  # Map tokens to their respective word.
    # new_tokens = tokenized_inputs.tokens(
    #     batch_index=0
    # )  # Map tokens to their respective word.
    # print("word_ids", word_ids)
    # print("new_tokens ", new_tokens)
    previous_word_idx: None | int = None
    sample_labels = example.get("ner_tags")
    if sample_labels is None:
        sample_labels = example.get("labels")
    if sample_labels is None:
        sample_labels = example.get("tags")
    assert sample_labels is not None
    assert background_label in labels

    assert isinstance(sample_labels[0], str)
    new_sample_labels = []
    for a in sample_labels:
        if a in labels:
            if a == background_label and phase == MachineLearningPhase.Training:
                new_sample_labels.append(-100)
            else:
                new_sample_labels.append(labels[a])
        else:
            assert phase != MachineLearningPhase.Training
            new_sample_labels.append(labels[background_label])
    sample_labels = new_sample_labels
    label_ids: list[int] = []
    for word_idx in word_ids:  # Set the special tokens to -100.
        if word_idx is None:
            label_ids.append(-100)
        elif (
            word_idx != previous_word_idx
        ):  # Only label the first token of a given word.
            # log_info("label is %s word_idx is %s", label, word_idx)
            label_ids.append(sample_labels[word_idx])
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx
    # print("labels are", label_ids)

    tokenized_inputs["labels"] = torch.tensor(label_ids, dtype=torch.long)
    return tokenized_inputs


# def collect_label(data: Any) -> Any:
#     assert len(data["ner_tags"]) == len(data["input_ids"])
#     data["labels"] = data.pop("ner_tags")
#     return data


def apply_tokenizer_transforms(
    dc: DatasetCollection,
    model_evaluator: HuggingFaceModelEvaluator,
) -> None:
    if not isinstance(model_evaluator.tokenizer, transformers.PreTrainedTokenizerBase):
        return

    tokenizer_kwargs = {
        "return_tensors": "pt",
    }
    input_max_len: int | None = dc.dataset_kwargs.get("input_max_len", None)
    if input_max_len is not None:
        log_info("use input text max_len %s", input_max_len)
        tokenizer_kwargs |= {
            "truncation": True,
            "padding": True,
            "max_length": input_max_len,
        }
    if model_evaluator.model_type == ModelType.TokenClassification:
        assert input_max_len is not None
        tokenizer_kwargs["padding"] = "max_length"
        labels = {
            label: idx
            for idx, label in enumerate(sorted(set(model_evaluator.model.labels)))
        }
        dc.append_named_transform(
            Transform(
                fun=functools.partial(
                    tokenize_and_align_labels,
                    model_evaluator.tokenizer,
                    tokenizer_kwargs,
                    labels,
                    MachineLearningPhase.Training,
                    "O",
                ),
                cacheable=True,
            ),
            phases=[MachineLearningPhase.Training],
        )
        dc.append_named_transform(
            Transform(
                fun=functools.partial(
                    tokenize_and_align_labels,
                    model_evaluator.tokenizer,
                    tokenizer_kwargs,
                    labels,
                    MachineLearningPhase.Test,
                    "O",
                ),
                cacheable=True,
            ),
            phases=[MachineLearningPhase.Validation, MachineLearningPhase.Test],
        )
        dc.append_named_transform(
            BatchTransform(fun=merge_batch_encodings, cacheable=True)
        )
        return
    dc.append_named_transform(
        BatchTransform(
            fun=functools.partial(
                model_evaluator.tokenizer,
                # nested_batch_encoding=model_evaluator.model_type == ModelType.CausalLM,
                **tokenizer_kwargs,
            ),
            component="input",
        )
    )
    if model_evaluator.model_type == ModelType.CausalLM:
        dc.append_named_transform(
            BatchTransform(
                name="DataCollatorForLanguageModeling",
                fun=functools.partial(
                    transformers.DataCollatorForLanguageModeling(
                        tokenizer=model_evaluator.tokenizer,
                        return_tensors="pt",
                        mlm=False,
                    )
                ),
                component="input",
            )
        )
        return
    dc.append_named_transform(
        BatchTransform(
            fun=functools.partial(
                transformers.DataCollatorWithPadding(
                    tokenizer=model_evaluator.tokenizer, **tokenizer_kwargs
                )
            ),
            component="input",
        )
    )


def huggingface_data_extraction(model_type: ModelType, data: Any) -> dict:
    if model_type in (ModelType.TokenClassification,):
        match data:
            case {"data": data, "index": index}:
                data.pop("id", None)
                return {"index": index, "input": data}
    return data


def add_text_extraction(dc: DatasetCollection, model_evaluator: Any) -> None:
    dc.append_named_transform(
        Transform(
            fun=functools.partial(
                huggingface_data_extraction, model_evaluator.model_type
            ),
            cacheable=True,
        )
    )

    dataset_name: str = dc.name.lower()
    # InputText
    if dataset_name == "imdb":
        dc.append_named_transform(
            Transform(
                fun=functools.partial(replace_str, old="<br />", new=""),
                component="input",
                cacheable=True,
            ),
        )
    if isinstance(dc, TextDatasetCollection):
        text_pipeline = dc.get_text_pipeline(model_evaluator.tokenizer)
        if text_pipeline is not None:
            for t in text_pipeline.transforms:
                dc.append_named_transform(t)


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
    add_text_extraction(dc=dc, model_evaluator=model_evaluator)
    if os.getenv("NO_TOKENIZER_TRANSFORMS") is None:
        apply_tokenizer_transforms(dc=dc, model_evaluator=model_evaluator)
