from typing import Tuple, Sequence, Dict, List, overload

import logging
import torch
from torch import Tensor
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
from torchvision.ops import box_convert

from .target import Target
from .formating import target_reset_tvtensor, target_box_format_to_enum, target_enum_to_box_format


def target_get_boxarea(target: Target) -> Tensor:
    if target["box_format"] == 0:
        area = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
    else:
        area = target["boxes"][:, 2] * target["boxes"][:, 3]
    return area

def target_normalize(target: Target, out_fmt: BoundingBoxFormat) -> Target:
    if target["boxes"].shape[0] == 0:
        return target
    h, w = target["size"]
    if out_fmt != target_box_format_to_enum(target["box_format"]).value:
        target["boxes"] = box_convert(target["boxes"],
                                      in_fmt=target_box_format_to_enum(target["box_format"]).value.lower(),
                                      out_fmt=out_fmt.name.lower()).float()
        target["box_format"] = target_enum_to_box_format(out_fmt)
        
    target["boxes"] = target["boxes"] / torch.tensor([w, h, w, h], dtype=torch.float32)
    
    return target

def target_create_empty(image_size: Tuple[int, int], # (H, W)
                        image_id: int | None = None,
                        attributes: List[str] | None = None) -> Target:
    return {
        "image_id": torch.tensor(image_id) if image_id is not None else torch.tensor(0),
        "orig_size": torch.tensor(image_size),
        "size": torch.tensor(image_size),
        "boxes": BoundingBoxes(torch.empty((0, 4)), format=BoundingBoxFormat.XYXY, canvas_size=image_size), # type: ignore[call-overload]
        "labels": torch.empty((0,), dtype=torch.int64),
        "box_format": torch.tensor(0),
        "attributes": {key: torch.empty((0,), dtype=torch.float32) for key in attributes} if attributes is not None else None,
    }
    
def target_filter(target: Target, keep: Tensor) -> Target:
    image_properties = ["image_id", "orig_size", "size", "box_format"]
    
    for key, value in target.items():
        if key in image_properties:
            continue
        elif key == "attributes":
            if value is None:
                continue
            for k, v in value.items():
                value[k] = v[keep]
        else:
            assert isinstance(value, Tensor)
            target[key] = value[keep]
        
    return target_reset_tvtensor(target)
    
@overload
def target_get(target: Target, key: str) -> Tensor | None:
    ...
@overload
def target_get(target: Target, key: List[str]) -> Dict[str, Tensor | None] | None:
    ...
def target_get(target: Target, key: str | List[str]) -> Tensor | Dict[str, Tensor | None] | None:
    attributes = target.get("attributes", None)
    out = None
    if attributes is not None:
        if isinstance(key, str):
            out = attributes.get(key, None)
        else:
            out = {k: attributes.get(k, None) for k in key}
    return None
    
def target_add_annotation(target: Target, category_id: int, bbox: Tensor | Sequence, attributes: Dict[str, Tensor], **kwargs) -> Target:
    if isinstance(bbox, list):
        bbox = torch.tensor(bbox)
    assert isinstance(bbox, Tensor)
    target["boxes"] = torch.cat((target["boxes"], bbox.unsqueeze(0)))
    target["labels"] = torch.cat((target["labels"], torch.tensor([category_id])))
    properties_exist = kwargs.get("attributes", None)
    if properties_exist is not None and attributes is not None:
        for key, value in attributes.items():
            if key in properties_exist:
                properties_exist[key] = torch.cat((properties_exist[key], value.unsqueeze(0)))
            else:
                logging.warning(f"Property {key} not found in existing attributes of target. Skipping.")
       
    target = target_reset_tvtensor(target)
    return target


