import torch
from torch import Tensor
from .target import Target
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
from torchvision.ops import box_convert

def target_box_format_to_enum(format: Tensor | int) -> BoundingBoxFormat:
    if isinstance(format, Tensor):
        format = int(format.item())
    
    if format == 0:
        return BoundingBoxFormat.XYXY
    elif format == 1:
        return BoundingBoxFormat.XYWH
    elif format == 2:
        return BoundingBoxFormat.CXCYWH
    else:
        raise ValueError(f"Unknown box format: {format}")
    
def target_enum_to_box_format(format: BoundingBoxFormat) -> Tensor:
    if format == BoundingBoxFormat.XYXY:
        return torch.tensor(0)
    elif format == BoundingBoxFormat.XYWH:
        return torch.tensor(1)
    elif format == BoundingBoxFormat.CXCYWH:
        return torch.tensor(2)

def target_reset_tvtensor(target: Target) -> Target:
    if isinstance(target['boxes'], BoundingBoxes):
        target["size"] = torch.tensor(target["boxes"].canvas_size)
        target["box_format"] = target_enum_to_box_format(target["boxes"].format)
    elif isinstance(target["boxes"], Tensor):
        canvas_size = (target["size"][0].item(), target["size"][1].item())
        target["boxes"] = BoundingBoxes(target["boxes"], format=target_box_format_to_enum(target["box_format"]), canvas_size=canvas_size) # type: ignore[call-overload]
        
    return target

def target_set_dtype(target: Target) -> Target:
    target["image_id"] = target["image_id"].to(torch.int64)
    target["size"] = target["size"].to(torch.int64)
    target["orig_size"] = target["orig_size"].to(torch.int64)
    target["boxes"] = target["boxes"].to(torch.float32)
    target["labels"] = target["labels"].to(torch.int64)
    target["box_format"] = target["box_format"].to(torch.int64)
    
    return target_reset_tvtensor(target)

def target_check_dtype(target: Target) -> bool:
    check_dict = {"image_id": torch.int64,
                  "size": torch.int64,
                  "orig_size": torch.int64,
                  "boxes": torch.float32,
                  "labels": torch.int64,
                  "box_format": torch.int64}
    for key, value in check_dict.items():
        if target[key].dtype != value:
            return False
    return True

def target_to_device(target: Target,
                     device: torch.device | str,
                     non_blocking = False,
                     ) -> Target:
    for key, value in target.items():
        if isinstance(value, Tensor):
            target[key] = value.to(device, non_blocking=non_blocking)
        elif isinstance(value, dict) and key == "attributes":
            for attr_key, attr_value in value.items():
                target[key][attr_key] = attr_value.to(device, non_blocking=non_blocking)
    
    return target