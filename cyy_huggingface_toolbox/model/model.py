import functools
import os
from typing import Callable, Any

import transformers
from cyy_naive_lib.log import log_warning
from cyy_torch_toolbox import ModelType


def __get_cache_dir() -> str:
    cache_dir = os.getenv("PYTORCH_MODEL_CACHE_ROOT_DIR")
    if cache_dir is None:
        return os.path.join(os.path.expanduser("~"), "huggingface_models")
    return cache_dir


def __create_huggingface_model(
    transformers_module: Any,
    model_name: str,
    pretrained: bool,
    **model_kwargs,
) -> Callable:
    if "cache_dir" not in model_kwargs:
        model_kwargs["cache_dir"] = __get_cache_dir()
    if pretrained:
        return transformers_module.from_pretrained(model_name, **model_kwargs)
    log_warning("use huggingface without pretrained parameters")
    config = transformers.AutoConfig.from_pretrained(model_name, **model_kwargs)
    model = transformers_module.from_config(config)
    return model


def get_huggingface_constructor(
    model_name: str,
) -> tuple[Callable, str, ModelType] | None:
    prefix_to_module = [
        (
            "hugging_face_sequence_classification_",
            transformers.AutoModelForSequenceClassification,
            ModelType.Classification,
        ),
        (
            "hugging_face_seq2seq_lm_",
            transformers.AutoModelForSeq2SeqLM,
            ModelType.Seq2SeqLM,
        ),
        (
            "hugging_face_token_classification_",
            transformers.AutoModelForTokenClassification,
            ModelType.TokenClassification,
        ),
        (
            "hugging_face_causal_lm_",
            transformers.AutoModelForCausalLM,
            ModelType.CausalLM,
        ),
    ]
    for prefix, transformers_module, model_type in prefix_to_module:
        if model_name.startswith(prefix):
            real_name = model_name[len(prefix) :]
            return (
                functools.partial(
                    __create_huggingface_model,
                    transformers_module,
                    real_name,
                ),
                real_name,
                model_type,
            )
    return None
