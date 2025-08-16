import os
import torch

from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM
from vllm import LLM


def merge_peft_model_for_vllm(
    pretrained_model_name_or_path: str,
    finetuned_model_dir: str,
) -> str:
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
    finetuned_model = PeftModel.from_pretrained(
        model=model, model_id=finetuned_model_dir
    )
    model = finetuned_model.merge_and_unload()
    saved_model_path = os.path.abspath(os.path.join(os.path.curdir, "finetuned_model"))
    assert not os.path.exists(saved_model_path)
    model.save_pretrained(saved_model_path)
    return saved_model_path


def get_vllm(
    pretrained_model_name_or_path: str, finetuned_model_dir: str | None = None, **kwargs
) -> LLM:
    model_name = pretrained_model_name_or_path
    if finetuned_model_dir is not None:
        pretrained_model_name_or_path = merge_peft_model_for_vllm(
            pretrained_model_name_or_path, finetuned_model_dir
        )
    if "tensor_parallel_size" not in kwargs:
        if torch.cuda.is_available():
            kwargs["tensor_parallel_size"] = torch.cuda.device_count()
    if "gpu_memory_utilization" not in kwargs:
        kwargs["gpu_memory_utilization"] = 0.5
    llm = LLM(
        model=pretrained_model_name_or_path,
        tokenizer=model_name,
        dtype="bfloat16",
        **kwargs,
    )
    llm.get_tokenizer().padding_side = "left"
    return llm
