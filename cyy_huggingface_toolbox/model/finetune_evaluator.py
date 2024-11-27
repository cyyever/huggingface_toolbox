from typing import Any

from peft.mapping import get_peft_model
from cyy_torch_toolbox import ModelType
from transformers import PreTrainedModel
from peft.utils.peft_types import TaskType
from peft import LoraConfig

from .evaluator import HuggingFaceModelEvaluator


class HuggingFaceModelEvaluatorForFinetune(HuggingFaceModelEvaluator):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        model = self.model
        assert isinstance(model, PreTrainedModel)
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
        self.set_model(get_peft_model(model=model, peft_config=peft_config))
