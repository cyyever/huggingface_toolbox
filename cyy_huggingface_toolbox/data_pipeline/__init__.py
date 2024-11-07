from cyy_torch_toolbox import DatasetType
from typing import Any
from cyy_torch_toolbox.data_pipeline import global_data_transform_factory
from .transform import (
    apply_tokenizer_transforms,
    add_text_extraction,
    add_text_transforms,
)


def append_transforms_to_dc(dc, model_evaluator: Any = None) -> None:
    add_text_extraction(dc=dc, model_evaluator=model_evaluator)
    if model_evaluator is not None:
        add_text_transforms(dc=dc, model_evaluator=model_evaluator)


global_data_transform_factory.register(DatasetType.Text, append_transforms_to_dc)

__all__ = ["apply_tokenizer_transforms"]
