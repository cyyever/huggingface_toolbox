import os

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
    pretrained_model_name_or_path: str, finetuned_model_dir: str | None = None
) -> LLM:
    model_name = pretrained_model_name_or_path
    if finetuned_model_dir is not None:
        pretrained_model_name_or_path = merge_peft_model_for_vllm(
            pretrained_model_name_or_path, finetuned_model_dir
        )
    llm = LLM(
        model=pretrained_model_name_or_path,
        generation_config="auto",
        tokenizer=model_name,
        dtype="bfloat16",
    )
    llm.get_tokenizter().padding_side = "left"
    return llm
