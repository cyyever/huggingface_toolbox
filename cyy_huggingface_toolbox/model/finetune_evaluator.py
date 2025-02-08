import copy
from typing import Any

import torch.nn
from cyy_torch_toolbox import ModelType, TensorDict
from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.tuners.lora.config import LoraConfig
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
            target_modules=kwargs.pop("finetune_modules"),
            task_type=TaskType.CAUSAL_LM,
        )
        self.set_model(get_peft_model(model=self.model, peft_config=peft_config))

    @property
    def peft_model(self) -> PeftModel:
        _model = self.model
        assert isinstance(_model, PeftModel)
        return _model

    @classmethod
    def get_perf_model_state_dict(cls, model: torch.nn.Module) -> TensorDict:
        assert isinstance(model, PeftModel)
        return get_peft_model_state_dict(model)
        # with TempDir() as dir_path:
        #     model.save_pretrained(dir_path)
        #     pool = TorchProcessPool()
        #     pool.submit(
        #         lambda dir_path: get_peft_model_state_dict(
        #             AutoPeftModelForCausalLM.from_pretrained(dir_path)
        #         ),
        #         dir_path,
        #     )
        #     res, _ = pool.wait_results()
        #     assert len(res) == 1
        #     return list(res.values())[0]

    def load_perf_model_state_dict(self, state_dict: TensorDict) -> None:
        set_peft_model_state_dict(
            model=self.peft_model, peft_model_state_dict=state_dict
        )

    def load_model_for_inference(self, model: torch.nn.Module) -> None:
        if self.model is model:
            self.set_model(copy.copy(model))
        self.load_perf_model_state_dict(
            HuggingFaceModelEvaluatorForFinetune.get_perf_model_state_dict(model)
        )
