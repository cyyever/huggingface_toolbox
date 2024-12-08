import functools
from typing import Any

import transformers
from cyy_naive_lib.log import log_info
from cyy_torch_toolbox import (
    DatasetCollection,
    ModelType,
    TextDatasetCollection,
    Transform,
    BatchTransform,
)
from cyy_torch_toolbox.data_pipeline.common import (
    replace_str,
)

from ..model import HuggingFaceModelEvaluator
from ..tokenizer import HuggingFaceTokenizer


def tokenize_and_align_labels(
    tokenizer: transformers.PreTrainedTokenizerFast, example
) -> transformers.BatchEncoding:
    tokenized_inputs = tokenizer(
        example["tokens"], padding=False, truncation=False, is_split_into_words=True
    )

    word_ids = tokenized_inputs.word_ids(
        batch_index=0
    )  # Map tokens to their respective word.
    previous_word_idx: None | int = None
    label = example["ner_tags"]
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
) -> None:
    if not isinstance(model_evaluator.tokenizer, HuggingFaceTokenizer):
        return

    input_max_len: int | None = dc.dataset_kwargs.get("input_max_len", None)
    if input_max_len is not None:
        log_info("use input text max_len %s", input_max_len)

    tokenizer_kwargs = {
        "padding": True,
        "max_length": input_max_len,
        "truncation": True,
        "return_tensors": "pt",
    }
    if model_evaluator.model_type == ModelType.TokenClassification:
        dc.append_named_transform(
            Transform(
                fun=functools.partial(
                    tokenize_and_align_labels, model_evaluator.tokenizer.tokenizer
                ),
                cacheable=True,
            )
        )
        dc.append_named_transform(
            BatchTransform(
                fun=functools.partial(
                    transformers.DataCollatorForTokenClassification(
                        tokenizer=model_evaluator.tokenizer.tokenizer
                    )
                ),
            )
        )
        return
    dc.append_named_transform(
        BatchTransform(
            fun=functools.partial(
                model_evaluator.tokenizer,
                nested_batch_encoding=model_evaluator.model_type == ModelType.CausalLM,
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
                        tokenizer=model_evaluator.tokenizer.tokenizer,
                        return_tensors="pt",
                        mlm=False,
                    )
                ),
            )
        )
        return
    tokenizer_kwargs.pop("truncation")
    dc.append_named_transform(
        BatchTransform(
            fun=functools.partial(
                transformers.DataCollatorWithPadding(
                    tokenizer=model_evaluator.tokenizer.tokenizer, **tokenizer_kwargs
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
        text_pipeline = dc.get_text_pipeline()
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
    apply_tokenizer_transforms(dc=dc, model_evaluator=model_evaluator)
