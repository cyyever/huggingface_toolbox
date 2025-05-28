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
        self.__tokenizer: HuggingFaceTokenizer = kwargs.pop("tokenizer", None)

    @property
    def tokenizer_mixin(self) -> HuggingFaceTokenizer:
        return self.__tokenizer

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizerFast:
        return self.__tokenizer.tokenizer

    def save_pretrained(self, output_dir: str) -> None:
        self.tokenizer.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)

    def set_tokenizer(self, tokenizer: HuggingFaceTokenizer) -> None:
        self.__tokenizer = tokenizer

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
        *,
        inputs: dict,
        **kwargs: Any,
    ) -> dict:
        match self.model_type:
            case ModelType.CausalLM:
                for v in inputs.values():
                    if isinstance(v, torch.Tensor):
                        v.squeeze_(dim=0)
                assert "labels" in inputs
            case ModelType.Classification:
                if "targets" in kwargs:
                    inputs["labels"] = kwargs["targets"]
            case ModelType.TokenClassification:
                assert inputs is None
                inputs = kwargs
        return inputs

    def get_feature_forward_fun(self) -> str:
        return "_forward_model"

    def generate(self, **kwargs: Any) -> list[str]:
        model_input = self._create_input(**kwargs)
        generated_ids = self.model.generate(
            **model_input,
            **(kwargs["generate_kwargs"]),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_texts = []
        for sample_input_ids, sample_generated_ids in zip(
            model_input["input_ids"], generated_ids, strict=False
        ):
            generated_texts.append(
                self.tokenizer.decode(
                    sample_generated_ids[sample_input_ids.shape[0] :],
                    skip_special_tokens=True,
                )
            )
        return generated_texts

    def _forward_model(
        self,
        *,
        device: torch.device | None = None,
        evaluation_mode: Any = None,
        batch_index: Any = None,
        batch_size: Any = None,
        non_blocking: bool | None = None,
        phase: Any = None,
        **kwargs: Any,
    ) -> dict:
        if "generate" in kwargs:
            return {"output": self.generate(**kwargs)}
        if "input_ids" in kwargs:
            kwargs["input_ids"] = kwargs["input_ids"].to(
                device=device, non_blocking=non_blocking
            )
        print(kwargs)
        model_input = self._create_input(**kwargs)
        output = self.model(**model_input)
        return self._compute_loss(
            **model_input, **output, evaluation_mode=evaluation_mode
        )

    def _compute_loss(self, **kwargs: Any) -> dict:
        ignore_index: int = -100
        assert kwargs.pop("reduce_loss", True)
        loss = kwargs.pop("loss", None)
        # HF computes loss in model forward
        if loss is None:
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
                    (kwargs["labels"][..., 1:] != ignore_index).sum().detach()
                )
            else:
                res["loss_batch_size"] = (
                    (kwargs["labels"].view(-1) != ignore_index).sum().detach()
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
