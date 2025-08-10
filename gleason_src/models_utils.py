import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Literal

from .pspnet import PSPNet
from .speed import set_speed_flags
from .preprocessing import CLASS_NAMES

class FastNormalise(nn.Module):
    def __init__(self, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), use_fp16=True):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor(std,  dtype=torch.float32).view(1,3,1,1))
        self.use_fp16 = use_fp16

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        if self.use_fp16 and x.is_cuda:
            return (x.half() - self.mean.half()) / self.std.half()
        return (x - self.mean) / self.std

class WrappedPSP(nn.Module):
    def __init__(self, net: nn.Module, img_size: int = 512):
        super().__init__()
        self.net = net
        self.norm = FastNormalise()
        self.img_size = img_size

    def forward(self, x, **kwargs):
        if x.ndim == 4 and x.shape[-1] == 3:  
            x = x.permute(0,3,1,2).contiguous(memory_format=torch.channels_last)

        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        
        x = self.norm(x)
        y = self.net(x, **kwargs)
        return y
        
    def monte_carlo_predict(self, x, n_samples=5, aux_weight=0.0):
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0,3,1,2).contiguous(memory_format=torch.channels_last)

        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        x = self.norm(x)

        return self.net.monte_carlo_predict(x, n_samples=n_samples, aux_weight=aux_weight)

def build_model(
    backbone: str = "resnet152",
    dropout: float = 0.1,
    img_size: int = 512,
    num_classes: int = 5,
    mode: Literal["compiled", "uncompiled"] = "uncompiled"
):
    set_speed_flags()

    base = PSPNet(num_classes=num_classes, backbone=backbone, dropout_rate=dropout)
    if hasattr(base, "model") and hasattr(base.model, "encoder") and hasattr(base.model.encoder, "set_grad_checkpointing"):
        base.model.encoder.set_grad_checkpointing(False)
    
    base = base.to(memory_format=torch.channels_last)
    
    if mode == "compiled":
        base = torch.compile(base, mode="reduce-overhead", dynamic=True)
    elif mode != "uncompiled":
        raise ValueError(f"Invalid mode '{mode}'. Must be 'compiled' or 'uncompiled'.")

    wrapped = WrappedPSP(base, img_size=img_size).to(memory_format=torch.channels_last).cuda()
    return wrapped

def ensure_num_classes(config: dict) -> dict:
    config = dict(config)
    config.setdefault("num_classes", len(CLASS_NAMES))
    return config