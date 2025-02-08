import functools
from collections.abc import Callable
from typing import Any

import torch
import transformers
from cyy_torch_toolbox import EvaluationMode, ModelEvaluator, ModelType
from transformers.loss.loss_utils import LOSS_MAPPING

from ..tokenizer import HuggingFaceTokenizer


class HuggingFaceModelEvaluator(ModelEvaluator):
    def __init__(self, model, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self.tokenizer: HuggingFaceTokenizer = kwargs.pop("tokenizer", None)

    def split_batch_input(self, inputs: Any, **kwargs: Any) -> dict:
        batch_dim = 0
        new_inputs = []
        batch_size: int = kwargs["batch_size"]
        if isinstance(inputs, transformers.BatchEncoding):
            assert "input_ids" in inputs
            for i in range(batch_size):
                sample = {}
                for k, v in inputs.items():
                    assert isinstance(v, torch.Tensor)
                    assert v.shape[0] == batch_size
                    sample[k] = v[i].unsqueeze(0)
                new_inputs.append(sample)
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
        **kwargs: Any,
    ) -> dict:
        if self.model_type in (ModelType.CausalLM,):
            for v in inputs.values():
                if isinstance(v, torch.Tensor):
                    v.squeeze_(dim=0)
            assert "labels" in inputs
        if self.model_type in (
            ModelType.Classification,
            ModelType.TokenClassification,
        ):
            inputs["labels"] = targets
        else:
            assert targets is None
        return inputs

    def get_feature_forward_fun(self) -> str:
        return "_forward_model"

    def generate(self, *args: Any, **kwargs: Any) -> list[str]:
        model_input = self._create_input(*args, **kwargs)
        generated_ids = self.model.generate(
            **model_input,
            **kwargs.pop("generate_kwargs", {}),
            pad_token_id=self.tokenizer.tokenizer.eos_token_id,
        )
        generated_texts = []
        for sample_input_ids, sample_generated_ids in zip(
            model_input["input_ids"], generated_ids
        ):
            generated_texts.append(
                self.tokenizer.tokenizer.decode(
                    sample_generated_ids[len(sample_input_ids)],
                    skip_special_tokens=True,
                )
            )
        return generated_texts

    def _forward_model(self, *args: Any, **kwargs: Any) -> dict:
        if kwargs.pop("generate", None) is not None:
            return {"output": self.generate(*args, **kwargs)}
        model_input = self._create_input(*args, **kwargs)
        output = self.model(**model_input)
        return self._compute_loss(
            **model_input, **output, evaluation_mode=kwargs["evaluation_mode"]
        )

    def _compute_loss(self, **kwargs: Any) -> dict:
        assert kwargs.pop("reduce_loss", True)
        if "pooled_logits" not in kwargs:
            kwargs["pooled_logits"] = kwargs["logits"]
        loss = self.loss_fun(**kwargs)

        res = {
            "loss": loss,
            "is_averaged_loss": True,
        }

        if kwargs["evaluation_mode"] != EvaluationMode.SampleInference:
            if self.model_type == ModelType.CausalLM:
                res["loss_batch_size"] = (
                    (kwargs["labels"][..., 1:] != -100).sum().detach()
                )
            else:
                res["loss_batch_size"] = (
                    (kwargs["labels"].view(-1) != -100).sum().detach()
                )
        if "logits" in kwargs:
            res["logits"] = kwargs["logits"]
        return res

    def _choose_loss_function_type(self) -> None | type:
        return None

    def _choose_loss_function(self) -> Callable:
        loss_type = ""
        kwargs = {}
        match self.model_type:
            case ModelType.Classification:
                loss_type = "ForSequenceClassification"
                kwargs["config"] = self.model.config
            case ModelType.TokenClassification:
                loss_type = "ForTokenClassification"
                kwargs["config"] = self.model.config
            case ModelType.CausalLM:
                loss_type = "ForCausalLM"
                kwargs["vocab_size"] = len(self.tokenizer.get_vocab())

        # Copied from hugging face
        return functools.partial(LOSS_MAPPING[loss_type], **kwargs)
