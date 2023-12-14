import sys
if sys.version_info < (3, 11):
    from typing_extensions import NotRequired, Required
else:
    from typing import NotRequired, Required
from typing import TypedDict
from torch import Tensor
from torchvision import tv_tensors

class Target(TypedDict):
    image_id:  Required[Tensor | tv_tensors.Image]
    size: Required[Tensor] # H, W
    orig_size: Required[Tensor] # H, W
    boxes: tv_tensors.BoundingBoxes | Tensor
    box_format: Tensor
    labels: Tensor
    attributes: NotRequired[dict[str, Tensor] | None]

