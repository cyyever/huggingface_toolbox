from typing import Any

import torch.nn
from cyy_torch_toolbox import ModelType
from peft import LoraConfig
from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.utils.peft_types import TaskType
from peft.utils.save_and_load import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from .evaluator import HuggingFaceModelEvaluator


class HuggingFaceModelEvaluatorForFinetune(HuggingFaceModelEvaluator):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert self.model_type == ModelType.CausalLM
        peft_config = LoraConfig(
            target_modules=[
                "q_proj",
                "o_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type=TaskType.CAUSAL_LM,
        )
        self.set_model(get_peft_model(model=self.model, peft_config=peft_config))

    def peft_model(self) -> PeftModel:
        _model = self.model
        assert isinstance(_model, PeftModel)
        return _model

    def load_model(self, model: torch.nn.Module) -> None:
        if self.model is model:
            return
        set_peft_model_state_dict(
            model=self.peft_model,
            peft_model_state_dict=get_peft_model_state_dict(model=model),
        )
