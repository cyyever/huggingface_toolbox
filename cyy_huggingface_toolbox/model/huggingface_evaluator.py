from typing import Any, Callable
import re

import functools
import torch
import transformers
from transformers.loss.loss_utils import LOSS_MAPPING
from cyy_torch_toolbox import ModelEvaluator, ModelType, Tokenizer
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class HuggingFaceModelEvaluator(ModelEvaluator):
    def __init__(self, model, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self.tokenizer: Tokenizer = kwargs.pop("tokenizer", None)

    def split_batch_input(self, inputs: Any, *args: Any, **kwargs: Any) -> dict:
        batch_dim = 0
        new_inputs = []
        if isinstance(inputs, transformers.BatchEncoding | dict):
            first_value = next(iter(inputs.values()))
            assert isinstance(first_value, torch.Tensor)
            for i in range(first_value.size(dim=0)):
                new_inputs.append({k: v[i].unsqueeze(dim=0) for k, v in inputs.items()})
            return {"inputs": new_inputs, "batch_dim": batch_dim}
        return {"inputs": inputs, "batch_dim": batch_dim}

    def get_input_feature(
        self, inputs: transformers.BatchEncoding
    ) -> transformers.BatchEncoding:
        assert isinstance(inputs, transformers.BatchEncoding | dict)
        if "inputs_embeds" not in inputs:
            input_ids = inputs["input_ids"]
            if hasattr(self.model, "distilbert"):
                assert isinstance(input_ids, torch.Tensor)
                if len(list(input_ids.shape)) == 1:
                    input_ids = input_ids.unsqueeze(dim=0)
                embeddings = self.model.distilbert.embeddings(input_ids).detach()
            elif hasattr(self.model, "bert"):
                embeddings = self.model.get_input_embeddings()(input_ids).detach()
            else:
                raise NotImplementedError(self.model)
            inputs["inputs_embeds"] = embeddings
        inputs.pop("input_ids", None)
        return inputs

    def get_input_embedding(self, inputs: transformers.BatchEncoding) -> torch.Tensor:
        return self.get_input_feature(inputs)["inputs_embeds"]

    def _create_input(
        self,
        inputs: dict,
        targets: Any,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        inputs["labels"] = targets
        if hasattr(targets, "input_ids"):
            inputs["labels"] = targets.input_ids
        return inputs

    def get_feature_forward_fun(self) -> str:
        return "_forward_model"

    def _forward_model(self, *args: Any, **kwargs: Any) -> dict:
        model_input = self._create_input(*args, **kwargs)
        output = self.model(**model_input)
        assert kwargs.get("reduce_loss", True)
        # targets = kwargs["targets"]
        # if kwargs.get("reduce_loss", True):
        #     return {
        #         "model_input": model_input,
        #         "model_output": output,
        #         "logits": output.logits,
        #         "loss": output.loss,
        #         "is_averaged_loss": True,
        #         "loss_batch_size": targets.shape[0],
        #     }
        return self._compute_loss(*args, **output, **kwargs)

    def _choose_loss_function(self) -> Callable:
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
