from typing import Any

import torch.nn


# Copyied from torchvision
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str,
    gamma: float,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        reduction (string): ``'mean'`` | ``'sum'``
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    cpu_targets = targets
    targets = targets.to(inputs.device, non_blocking=True)
    mask = cpu_targets != ignore_index
    inputs = inputs[mask]
    targets = targets[mask]
    cpu_targets = cpu_targets[mask]
    ce_loss = torch.nn.functional.cross_entropy(
        inputs,
        targets,
        ignore_index=ignore_index,
        reduction="none",
    )
    p = torch.softmax(inputs, dim=1)
    assert p.shape == inputs.shape
    assert len(p.shape) == 2
    mask2 = torch.zeros_like(p, dtype=torch.bool)
    for idx, target in enumerate(cpu_targets):
        mask2[idx][target] = True
    p_t = torch.masked_select(p, mask2).flatten()
    assert p_t.shape == ce_loss.shape
    loss = ce_loss * ((1 - p_t) ** gamma)
    assert loss.shape[0] == targets.shape[0]

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'mean', 'sum'"
        )
    return loss


# copied from HuggingFace ForCausalLMLoss
def focal_loss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: int | None = None,
    ignore_index: int = -100,
    gamma: float = 2,
    **kwargs: Any,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    # Shift so that tokens < n predict n
    labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = sigmoid_focal_loss(
        inputs=logits,
        targets=shift_labels,
        reduction=reduction,
        gamma=gamma,
        ignore_index=ignore_index,
    )
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss
