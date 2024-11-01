def squeeze_huggingface_input(huggingface_input: dict) -> dict:
    for k in ("input_ids", "attention_mask"):
        if k in huggingface_input:
            huggingface_input[k] = huggingface_input[k].squeeze(dim=0)
    return huggingface_input


__all__ = ["squeeze_huggingface_input"]
