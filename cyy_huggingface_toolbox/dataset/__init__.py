from cyy_torch_toolbox import DatasetType
from cyy_torch_toolbox.dataset.repository import (
    register_dataset_factory,
)

from .repository import HuggingFaceFactory

register_dataset_factory(factory=HuggingFaceFactory(), dataset_type=DatasetType.Text)
