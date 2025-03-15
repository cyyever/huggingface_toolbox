from typing import Any
import torch.nn


# copied from HuggingFace ForCausalLMLoss
def focal_loss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: int | None = None,
    ignore_index: int = -100,
    **kwargs: Any,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = torch.nn.functional.cross_entropy(
        logits, shift_labels, ignore_index=ignore_index, reduction=reduction
    )
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss
