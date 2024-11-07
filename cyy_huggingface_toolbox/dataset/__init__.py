from .repository import HunggingFaceFactory

from cyy_torch_toolbox import DatasetType, DatasetUtil
from cyy_torch_toolbox.dataset.repository import (
    register_dataset_factory,
    global_dataset_util_factor,
)


global_dataset_util_factor.register(DatasetType.Text, DatasetUtil)
register_dataset_factory(DatasetType.Text, HunggingFaceFactory())
register_dataset_factory(DatasetType.CodeText, HunggingFaceFactory())
