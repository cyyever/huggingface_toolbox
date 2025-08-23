import copy
import functools
import os
from collections.abc import Callable
from typing import Any

import torch
import transformers
from cyy_naive_lib.log import log_error, log_warning
from cyy_torch_toolbox import ModelType
from peft.utils.other import prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig


def __create_huggingface_model(
    transformers_module: Any,
    model_name: str,
    pretrained: bool,
    **model_kwargs,
) -> Callable:
    try:
        model_kwargs = copy.deepcopy(model_kwargs)
        if "device_map" not in model_kwargs:
            model_kwargs["device_map"] = "cpu"
        model_kwargs["trust_remote_code"] = True
        model_kwargs.pop("loss_fun_name", None)
        model_kwargs.pop("loss_fun_kwargs", None)
        model_kwargs.pop("frozen_modules", None)
        use_gradient_checkpointing = model_kwargs.pop(
            "use_gradient_checkpointing", False
        )
        if pretrained or "finetune_modules" in model_kwargs:
            model_kwargs.pop("finetune_modules", None)
            model_kwargs.pop("finetune_config", None)
            bnb_config: BitsAndBytesConfig | None = None
            if model_kwargs.pop("load_in_4bit", False):
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="nf4",
                )
            elif model_kwargs.pop("load_in_8bit", False):
                model_kwargs.pop("load_in_8bit")
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            if bnb_config is not None:
                model_kwargs["quantization_config"] = bnb_config
                if "torch_dtype" not in model_kwargs:
                    model_kwargs["torch_dtype"] = torch.bfloat16
            # prepare_model_for_kbit_training_flag = model_kwargs.pop(
            #     "prepare_model_for_kbit_training", True
            # )
            model = transformers_module.from_pretrained(model_name, **model_kwargs)
            if bnb_config is not None:
                return prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=use_gradient_checkpointing
                )
            return model

        log_warning("use huggingface without pretrained parameters")
        config = transformers.AutoConfig.from_pretrained(model_name, **model_kwargs)
        model = transformers_module.from_config(config)
        return model
    except BaseException as e:
        log_error(
            "Failed to create huggingface model, that shouldn't happen."
            " model_kwargs is %s, exception is %s",
            model_kwargs,
            e,
        )
        raise e


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
        (
            "file",
            transformers.AutoModel,
            ModelType.UnknownType,
        ),
    ]
    for prefix, transformers_module, model_type in prefix_to_module:
        real_name: str | None = None
        if model_name.startswith(prefix):
            real_name = model_name.removeprefix(prefix)
        if prefix == "file":
            if os.path.exists(model_name):
                if os.path.isdir(model_name):
                    real_name = model_name
                    log_warning("load local huggingface model: %s", real_name)
                else:
                    log_warning(
                        "expect the local hugging face model to be a folder, but it is a file: %s",
                        model_name,
                    )
            elif model_name.startswith("/"):
                log_warning("not a path %s", model_name)
        if real_name is not None:
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
