import os, subprocess, torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import kornia
import kornia.augmentation as Kau
import numpy as np
from typing import Tuple, Optional 

class PSPNet(nn.Module):
    def __init__(self, num_classes=5, backbone='resnet50', dropout_rate=0.1, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.model = smp.PSPNet(
            encoder_name=backbone, 
            encoder_weights="imagenet", 
            in_channels=3, 
            classes=num_classes,
            aux_params={"dropout": dropout_rate, "classes": num_classes}
        ).to(memory_format=torch.channels_last)

        if hasattr(self.model.encoder, 'set_grad_checkpointing'):
            self.model.encoder.set_grad_checkpointing(False)

    @torch.amp.autocast('cuda', enabled=torch.cuda.is_available())
    def forward(self, x: torch.Tensor, aux_weight: float = 0.4) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if aux_weight > 1e-6:
            return self.model(x)
        else:
            feats = self.model.encoder(x)
            dec   = self.model.decoder(feats)
            masks = self.model.segmentation_head(dec)
            return masks, None

    def enable_monte_carlo_dropout(self):
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def monte_carlo_predict(self, x, n_samples=5, aux_weight=0.0):
        self.enable_monte_carlo_dropout()
        x = x.contiguous(memory_format=torch.channels_last)
        
        mean = None
        m2   = None
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            for t in range(n_samples):
                logits, _ = self.forward(x, aux_weight=0.0)
                p = F.softmax(logits, dim=1)
                if mean is None:
                    mean = p
                    m2   = torch.zeros_like(p)
                else:
                    delta = p - mean
                    mean  = mean + delta / (t + 1)
                    m2    = m2 + delta * (p - mean)
        
        var = m2 / max(n_samples - 1, 1)
        uncertainty = var.mean(dim=1)     
        return mean, uncertainty

class KorniaTrainAugmentation(nn.Module):
    def __init__(self, img_size: int, max_rotation: float = 20.0, max_colour: float = 0.3, max_blur: float = 0.5):
        super().__init__()
        self.img_size = img_size

        self.geometric_aug = Kau.AugmentationSequential(
            Kau.RandomHorizontalFlip(p=0.5),
            Kau.RandomVerticalFlip(p=0.5),
            Kau.RandomAffine(
                degrees=max_rotation,
                translate=0.1,
                scale=(0.85, 1.15),
                p=1.0,
                padding_mode='border',
                align_corners=False
            ),
            same_on_batch=False
        )

        self.colour_jitter = Kau.ColorJitter(
            brightness=max_colour, contrast=max_colour, saturation=max_colour, hue=0.15, p=0.7
        )
        self.blur = Kau.RandomGaussianBlur(kernel_size=(3, 9), sigma=(0.1, 2.0), p=max_blur)

    def _maximal_square_coords(self, mask_hw_bool: np.ndarray) -> Tuple[int, int, int]:
        H, W = mask_hw_bool.shape
        if mask_hw_bool.all():
            s = min(H, W)
            y, x = (H - s) // 2, (W - s) // 2
            return x, y, s

        m = mask_hw_bool.astype(np.uint8)
        dp = np.zeros((H + 1, W + 1), dtype=np.int32)
        max_s, max_i, max_j = 0, 0, 0
        for i in range(1, H + 1):
            for j in range(1, W + 1):
                if m[i - 1, j - 1]:
                    s = min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]) + 1
                    dp[i, j] = s
                    if s > max_s:
                        max_s, max_i, max_j = s, i, j
        if max_s == 0:
            return W // 2, H // 2, 1 
        return max_j - max_s, max_i - max_s, max_s

    @torch.no_grad()
    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if x.ndim == 4 and x.shape[-1] == 3:      
            x = x.permute(0, 3, 1, 2)
        x = x.to(torch.float32)
        if x.max().item() > 1.0:                  
            x = x.mul_(1.0 / 255.0)
        x = x.contiguous(memory_format=torch.channels_last)

        B, _, H, W = x.shape
        device = x.device

        params = self.geometric_aug.forward_parameters(x.shape)
        x = self.geometric_aug(x, params=params)
        M = self.geometric_aug.get_transformation_matrix(x, params)

        ones = torch.ones(B, 1, H, W, device=device, dtype=x.dtype)
        valid = kornia.geometry.transform.warp_perspective(
            ones, M, dsize=(H, W), padding_mode="zeros", align_corners=False
        ).squeeze(1)

        coords = [self._maximal_square_coords((valid[i] > 0.5).cpu().numpy()) for i in range(B)]
        src = torch.empty(B, 4, 2, device=device, dtype=torch.float32)
        for i, (x0, y0, s) in enumerate(coords):
            src[i] = torch.tensor([[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s]],
                                device=device, dtype=torch.float32)

        dst = torch.tensor([[[0, 0],
                            [self.img_size - 1, 0],
                            [self.img_size - 1, self.img_size - 1],
                            [0, self.img_size - 1]]],
                        device=device, dtype=torch.float32).expand(B, -1, -1)

        T = kornia.geometry.transform.get_perspective_transform(src, dst)
        x = kornia.geometry.transform.warp_perspective(
            x, T, dsize=(self.img_size, self.img_size),
            mode='bilinear', padding_mode='border', align_corners=False
        )

        self.colour_jitter.p = 0.7 * alpha
        self.blur.p = 0.5 * alpha
        x = self.colour_jitter(x.float())
        x = self.blur(x.float())

        return x.clamp_(0, 1).contiguous(memory_format=torch.channels_last)

class KorniaValidationAugmentation(nn.Module):
    def __init__(self, img_size: int):
        super().__init__()
        self.img_size = img_size

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[-1] == 3:          
            x = x.permute(0, 3, 1, 2).contiguous()
        x = x.to(torch.float32)
        if x.max().item() > 1.0:                       
            x = x.mul_(1.0 / 255.0)

        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode='bilinear', align_corners=False)

        return x.contiguous(memory_format=torch.channels_last)

for _Cls in (KorniaTrainAugmentation, KorniaValidationAugmentation):
    if not getattr(_Cls, "_dynamo_patched", False):
        if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "disable"):
            _Cls.forward = torch._dynamo.disable(_Cls.forward)
            _Cls._dynamo_patched = True