import functools
from typing import Callable

import transformers
from cyy_naive_lib.log import get_logger


def __create_huggingface_seq2seq_lm_model(
    model_name: str, pretrained: bool, **model_kwargs
):
    if pretrained:
        pretrained_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            model_name, **model_kwargs
        )
        return pretrained_model
    get_logger().warning("use huggingface without pretrained parameters")
    config = transformers.AutoConfig.from_pretrained(model_name, **model_kwargs)
    model = transformers.AutoModelForSeq2SeqLM.from_config(config)
    return model


def __create_huggingface_sequence_classification_model(
    model_name: str, pretrained: bool, **model_kwargs
):
    if pretrained:
        pretrained_model = (
            transformers.AutoModelForSequenceClassification.from_pretrained(
                model_name, **model_kwargs
            )
        )
        return pretrained_model
    get_logger().warning("use huggingface without pretrained parameters")
    config = transformers.AutoConfig.from_pretrained(model_name, **model_kwargs)
    model = transformers.AutoModelForSequenceClassification.from_config(config)
    return model


def __create_huggingface_model(model_name: str, pretrained: bool, **model_kwargs):
    if pretrained:
        pretrained_model = transformers.AutoModel.from_pretrained(
            model_name, **model_kwargs
        )
        return pretrained_model
    get_logger().warning("use huggingface without pretrained parameters")
    config = transformers.AutoConfig.from_pretrained(model_name, **model_kwargs)
    model = transformers.AutoModel.from_config(config)
    return model


def get_huggingface_constructor(model_name: str) -> tuple[Callable, str] | None:
    prefix = "hugging_face_sequence_classification_"
    if model_name.startswith(prefix):
        real_name = model_name[len(prefix) :]
        return (
            functools.partial(
                __create_huggingface_sequence_classification_model, real_name
            ),
            real_name,
        )
    prefix = "hugging_face_seq2seq_lm_"
    if model_name.startswith(prefix):
        real_name = model_name[len(prefix) :]
        return (
            functools.partial(__create_huggingface_seq2seq_lm_model, real_name),
            real_name,
        )
    prefix = "hugging_face_"
    if model_name.startswith(prefix):
        real_name = model_name[len(prefix) :]
        return functools.partial(__create_huggingface_model, real_name), real_name
    return None
