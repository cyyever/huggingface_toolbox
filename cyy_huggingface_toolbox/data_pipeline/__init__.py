import torch


def squeeze_huggingface_input(huggingface_input: dict) -> dict:
    for k, v in huggingface_input.items():
        if isinstance(v, torch.Tensor):
            huggingface_input[k] = huggingface_input[k].squeeze(dim=0)
    return huggingface_input


__all__ = ["squeeze_huggingface_input"]
