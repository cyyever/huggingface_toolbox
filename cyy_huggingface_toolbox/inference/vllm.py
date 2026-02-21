import os
import shutil
from collections.abc import Generator, Iterable
from typing import Any

import torch
from peft.peft_model import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    ProcessorMixin,
)
from vllm import LLM, RequestOutput, SamplingParams


def merge_peft_model_for_vllm(
    pretrained_model_name_or_path: str,
    finetuned_model_dir: str,
    processor: ProcessorMixin | None = None,
) -> str:
    try:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
    except BaseException:
        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path
        )
    finetuned_model = PeftModel.from_pretrained(
        model=model, model_id=finetuned_model_dir
    )
    model = finetuned_model.merge_and_unload()
    saved_model_path = os.path.abspath(
        os.path.join(finetuned_model_dir, "finetuned_model_for_vllm")
    )
    if os.path.exists(saved_model_path):
        assert os.path.isdir(saved_model_path)
        shutil.rmtree(saved_model_path)
    model.save_pretrained(saved_model_path)
    if processor is not None:
        processor.save_pretrained(saved_model_path)
    return saved_model_path


def get_llm_engine(
    pretrained_model_name_or_path: str,
    finetuned_model_dir: str | None = None,
    processor: ProcessorMixin | None = None,
    **kwargs: Any,
) -> LLM:
    model_name = pretrained_model_name_or_path
    if finetuned_model_dir is not None:
        pretrained_model_name_or_path = merge_peft_model_for_vllm(
            pretrained_model_name_or_path, finetuned_model_dir, processor=processor
        )
    if "tensor_parallel_size" not in kwargs and torch.cuda.is_available():
        kwargs["tensor_parallel_size"] = torch.cuda.device_count()
    if "gpu_memory_utilization" not in kwargs:
        kwargs["gpu_memory_utilization"] = 0.8
    if "dtype" not in kwargs:
        kwargs["dtype"] = "bfloat16"

    llm = LLM(
        model=pretrained_model_name_or_path,
        tokenizer=model_name,
        **kwargs,
    )
    tokenizer = llm.get_tokenizer()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return llm


def get_vllm_chat_output(
    llm: LLM, dataloader: Iterable[dict[str, Any]], index_key: str, sampling_params: SamplingParams | None = None
) -> Generator[tuple[RequestOutput, int | str]]:
    # Load the default sampling parameters from the model.
    if sampling_params is None:
        sampling_params = SamplingParams(n=1, max_tokens=2048, temperature=0)

    for batch in dataloader:
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        yield from zip(
            llm.chat(
                batch["conversations"],
                sampling_params=sampling_params,
            ),
            batch[index_key],
            strict=False,
        )
