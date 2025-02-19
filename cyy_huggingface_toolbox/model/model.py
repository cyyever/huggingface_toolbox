import copy
import functools
import os
from collections.abc import Callable
from typing import Any

import torch
import transformers
from cyy_naive_lib.log import log_warning
from cyy_torch_toolbox import ModelType
from peft.utils.other import prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig


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
    model_kwargs = copy.deepcopy(model_kwargs)
    if "device_map" not in model_kwargs:
        model_kwargs["device_map"] = "cpu"
    model_kwargs["trust_remote_code"] = True
    if "cache_dir" not in model_kwargs:
        model_kwargs["cache_dir"] = __get_cache_dir()
    if pretrained or "finetune_modules" in model_kwargs:
        model_kwargs.pop("finetune_modules",None)
        bnb_config: BitsAndBytesConfig | None = None
        if "load_in_4bit" in model_kwargs:
            model_kwargs.pop("load_in_4bit")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
            )
        elif "load_in_8bit" in model_kwargs:
            model_kwargs.pop("load_in_8bit")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["torch_dtype"] = torch.bfloat16
        model = transformers_module.from_pretrained(model_name, **model_kwargs)
        if bnb_config is not None:
            return prepare_model_for_kbit_training(model)
        return model

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
