from cyy_torch_toolbox.dataset.repository import (
    register_dataset_factory,
)

from .repository import HunggingFaceFactory

register_dataset_factory(factory=HunggingFaceFactory())
