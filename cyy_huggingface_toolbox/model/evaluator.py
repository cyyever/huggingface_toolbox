from typing import Any, Callable

import functools
import torch
import transformers
from transformers.loss.loss_utils import LOSS_MAPPING
from cyy_torch_toolbox import ModelEvaluator, Tokenizer, ModelType


class HuggingFaceModelEvaluator(ModelEvaluator):
    def __init__(self, model, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self.tokenizer: Tokenizer = kwargs.pop("tokenizer", None)

    def split_batch_input(self, inputs: Any, batch_size: int) -> dict:
        batch_dim = 0
        new_inputs = []
        if isinstance(inputs, transformers.BatchEncoding):
            for i in range(batch_size):
                new_inputs.append(inputs.sequence_ids(i))
            return {"inputs": new_inputs, "batch_dim": batch_dim}
        return {"inputs": inputs, "batch_dim": batch_dim}

    def get_input_feature(
        self, inputs: transformers.BatchEncoding
    ) -> transformers.BatchEncoding:
        assert isinstance(inputs, transformers.BatchEncoding | dict)
        if "inputs_embeds" not in inputs:
            input_ids = inputs["input_ids"]
            if isinstance(self.model, transformers.PreTrainedModel):
                embeddings = self.model.get_input_embeddings()(input_ids).detach()
            else:
                raise NotImplementedError(self.model)
            inputs["inputs_embeds"] = embeddings
        inputs.pop("input_ids", None)
        return inputs

    def get_input_embedding(self, inputs: transformers.BatchEncoding) -> torch.Tensor:
        res = self.get_input_feature(inputs)["inputs_embeds"]
        assert isinstance(res, torch.Tensor)
        return res

    def _create_input(
        self,
        inputs: dict,
        targets: Any,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        if self.model_type in (ModelType.Classification, ModelType.TokenClassification):
            inputs["labels"] = targets
            if hasattr(targets, "input_ids"):
                inputs["labels"] = targets.input_ids
        else:
            assert targets is None
        return inputs

    def get_feature_forward_fun(self) -> str:
        return "_forward_model"

    def _forward_model(self, *args: Any, **kwargs: Any) -> dict:
        model_input = self._create_input(*args, **kwargs)
        output = self.model(**model_input)
        return self._compute_loss(*args, **output, **kwargs)

    def _compute_loss(self, **kwargs: Any) -> dict:
        assert kwargs.pop("reduce_loss", True)
        targets = kwargs["targets"]
        if "labels" not in kwargs:
            kwargs["labels"] = kwargs["targets"]
        if "pooled_logits" not in kwargs:
            kwargs["pooled_logits"] = kwargs["logits"]
        loss = self.loss_fun(**kwargs)

        res = {
            "loss": loss,
            "is_averaged_loss": True,
            "loss_batch_size": (targets.view(-1) != -100).sum(),
        }
        if "logits" in kwargs:
            res["logits"] = kwargs["logits"]
        return res

    def _choose_loss_function(self) -> Callable:
        # Copied from hugging face
        if getattr(self.model.config, "loss_type", None) is not None:
            loss_type = self.model.config.loss_type
        else:
            loss_type = self.model.__class__.__name__
            if loss_type not in LOSS_MAPPING:
                for predefined_loss_type in LOSS_MAPPING:
                    if predefined_loss_type in loss_type:
                        loss_type = predefined_loss_type
                        break
        assert loss_type is not None
        return functools.partial(LOSS_MAPPING[loss_type], config=self.model.config)
