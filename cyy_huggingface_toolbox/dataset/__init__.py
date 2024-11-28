from cyy_torch_toolbox import DatasetType
from cyy_torch_toolbox.dataset.repository import (
    register_dataset_factory,
)

from .repository import HunggingFaceFactory

register_dataset_factory(factory=HunggingFaceFactory(), dataset_type=DatasetType.Text)
