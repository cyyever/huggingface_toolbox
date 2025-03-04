from typing import Any

import torch.nn
from cyy_naive_lib.log import log_debug, log_info
from cyy_torch_toolbox import ModelType, TensorDict, tensor_to
from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.tuners.lora.config import LoraConfig
from peft.utils.peft_types import TaskType
from peft.utils.save_and_load import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import PreTrainedModel

from .evaluator import HuggingFaceModelEvaluator


class HuggingFaceModelEvaluatorForFinetune(HuggingFaceModelEvaluator):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert self.model_type == ModelType.CausalLM
        log_info("Finetune modules %s", kwargs["finetune_modules"])
        finetune_config = kwargs.get("finetune_config", {})
        if finetune_config:
            log_info("Use finetune config %s", finetune_config)
        else:
            log_debug("Use default finetune config")
        config_type: type = LoraConfig
        assert "type" not in finetune_config

        peft_config = config_type(
            target_modules=kwargs["finetune_modules"],
            task_type=TaskType.CAUSAL_LM,
            **finetune_config,
        )
        log_info("Use peft config %s", peft_config)
        model = self.model
        assert isinstance(model, PreTrainedModel)
        self.set_model(get_peft_model(model=model, peft_config=peft_config))

    @property
    def peft_model(self) -> PeftModel:
        _model = self.model
        assert isinstance(_model, PeftModel)
        return _model

    @classmethod
    def get_perf_model_state_dict(cls, model: torch.nn.Module) -> TensorDict:
        assert isinstance(model, PeftModel)
        return get_peft_model_state_dict(model)

    def load_perf_model_state_dict(
        self, state_dict: TensorDict, device: torch.device
    ) -> None:
        state_dict = tensor_to(state_dict, device=device)
        _, unexpected_keys = set_peft_model_state_dict(
            model=self.peft_model,
            peft_model_state_dict=state_dict,
            ignore_mismatched_sizes=False,
        )
        assert not unexpected_keys

    def to_device(self, device: torch.device, non_blocking: bool = True) -> None:
        self.model_util.to_device(device=device, non_blocking=non_blocking)
        perf_model_state_dict = self.get_perf_model_state_dict(self.peft_model)
        self.load_perf_model_state_dict(state_dict=perf_model_state_dict, device=device)
