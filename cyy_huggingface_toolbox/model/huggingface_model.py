import functools
from typing import Callable, Any

import transformers
import transformers.models
from cyy_naive_lib.log import get_logger


def __create_huggingface_model(
    transformers_module: Any,
    model_name: str,
    pretrained: bool,
    **model_kwargs,
) -> Callable:
    model_kwargs["attn_implementation"] = "eager"
    if pretrained:
        return transformers_module.from_pretrained(model_name, **model_kwargs)
    get_logger().warning("use huggingface without pretrained parameters")
    config = transformers.AutoConfig.from_pretrained(model_name, **model_kwargs)
    model = transformers_module.from_config(config)
    return model


def get_huggingface_constructor(model_name: str) -> tuple[Callable, str] | None:
    prefix_to_module = [
        (
            "hugging_face_sequence_classification_",
            transformers.AutoModelForSequenceClassification,
        ),
        ("hugging_face_seq2seq_lm_", transformers.AutoModelForSeq2SeqLM),
        (
            "hugging_face_token_classification_",
            transformers.AutoModelForTokenClassification,
        ),
        ("hugging_face_", transformers.AutoModel),
    ]
    for prefix, transformers_module in prefix_to_module:
        if model_name.startswith(prefix):
            real_name = model_name[len(prefix) :]
            return (
                functools.partial(
                    __create_huggingface_model,
                    transformers_module,
                    real_name,
                ),
                real_name,
            )
    return None
