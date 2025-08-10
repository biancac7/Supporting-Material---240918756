import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Type, Tuple, Optional, Any, Callable
from itertools import combinations

from gleason_src.preprocessing import CLINICAL_CLASS_INDICES, CLASS_NAMES

LOSSES: Dict[str, Type[nn.Module]] = {}

def register_loss(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in LOSSES:
            raise ValueError(f"Loss with name '{name}' already registered.")
        LOSSES[name] = cls
        return cls
    return decorator

@register_loss('wce')
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, **kwargs: Any):
        super().__init__()
        self.register_buffer('weights', torch.as_tensor(class_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 4:
            log_probs = F.log_softmax(logits, dim=1)
            return -(targets * log_probs * self.weights.view(1, -1, 1, 1)).sum(dim=1)
        elif logits.ndim == 2:
            return F.cross_entropy(logits, targets, weight=self.weights, reduction='none')
        else:
            return torch.tensor(0.0, device=logits.device)

@register_loss('focal')
class FocalLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0, **kwargs: Any):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('alpha', torch.as_tensor(class_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 4:
            probs = F.softmax(logits, dim=1)
            log_probs = torch.log(probs + 1e-8)
            p_t = (targets * probs).sum(dim=1)
            focal_weight = (1.0 - p_t).pow(self.gamma)
            alpha_expanded = self.alpha.view(1, -1, 1, 1)
            ce_loss = -(targets * log_probs * alpha_expanded).sum(dim=1)
            return focal_weight * ce_loss
        elif logits.ndim == 2:
            B = logits.shape[0]
            log_probs = F.log_softmax(logits, dim=1)
            log_p_t = log_probs[torch.arange(B), targets]
            p_t = log_p_t.exp()
            alpha_t = self.alpha[targets]
            focal_loss = -alpha_t * (1 - p_t).pow(self.gamma) * log_p_t
            return focal_loss
        else:
            return torch.tensor(0.0, device=logits.device)

@register_loss('emd')
class EMDLoss(nn.Module):
    def __init__(self, cost_type: str = 'L2', **kwargs: Any):
        super().__init__()
        self.cost_type = cost_type

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_norm = targets / (targets.sum(dim=1, keepdim=True) + 1e-8)
        
        probs_cdf = torch.cumsum(probs, dim=1)
        targets_cdf = torch.cumsum(targets_norm, dim=1)

        if self.cost_type == 'L1':
            loss = torch.abs(probs_cdf - targets_cdf).sum(dim=1)
        else:
            loss = ((probs_cdf - targets_cdf) ** 2).sum(dim=1)
        
        return loss

@register_loss('williams_index')
class WilliamsIndexLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-6, **kwargs: Any):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        individual_masks = batch_data.get('individual_masks')
        if individual_masks is None or individual_masks.ndim != 4 or individual_masks.shape[1] < 2:
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            return 0.01 * entropy

        pred_probs = F.softmax(logits, dim=1)
        B, P, H, W = individual_masks.shape
        C = pred_probs.shape[1]

        if pred_probs.shape[0] != B:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        individual_masks_one_hot = F.one_hot(individual_masks.long().clamp(0, C - 1), num_classes=C).float()
        individual_masks_one_hot = individual_masks_one_hot.permute(0, 1, 4, 2, 3)

        pred_probs_expanded = pred_probs.unsqueeze(1)

        model_expert_errors_per_expert = torch.abs(pred_probs_expanded - individual_masks_one_hot).mean(dim=(0, 2, 3, 4))
        avg_model_expert_error = model_expert_errors_per_expert.mean()

        expert_expert_errors = []
        for p1, p2 in combinations(range(P), 2):
            expert1_probs = individual_masks_one_hot[:, p1]
            expert2_probs = individual_masks_one_hot[:, p2]

            error = torch.abs(expert1_probs - expert2_probs).mean()
            expert_expert_errors.append(error)

        if not expert_expert_errors:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        avg_expert_expert_error = torch.stack(expert_expert_errors).mean()

        inverse_model_expert = 1.0 / (avg_model_expert_error + self.epsilon)
        inverse_expert_expert = 1.0 / (avg_expert_expert_error + self.epsilon)

        williams_index = inverse_model_expert / (inverse_expert_expert + self.epsilon)

        return -torch.log(williams_index + self.epsilon)

class DisagreementWeighter(nn.Module):
    def __init__(self, metric: str, epsilon: float = 1e-6, **kwargs: Any):
        super().__init__()
        self.metric = metric
        self.epsilon = epsilon

    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        map_key = f'disagreement_map_{self.metric}'
        disagreement_map = batch_data.get(map_key)

        if disagreement_map is None:
             return 1.0

        disagreement_map = disagreement_map.squeeze(1)
        return 1.0 / (disagreement_map + self.epsilon)

class ScaledWeighter(nn.Module):
    def __init__(self, class_weights: torch.Tensor, priority_scale_factor: float):
        super().__init__()
        weights = class_weights.clone()
        for idx in CLINICAL_CLASS_INDICES:
            weights[idx] *= priority_scale_factor
        self.register_buffer('pixel_weights', weights)

    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        target_soft = batch_data['soft_label']
        target_hard = target_soft.argmax(dim=1)
        return self.pixel_weights[target_hard]

class MeanReducer(nn.Module):
    def forward(self, loss_map: torch.Tensor, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return loss_map.mean()

class PrioritySplitReducer(nn.Module):
    def __init__(self, priority_weight: float):
        super().__init__()
        self.priority_weight = priority_weight
        priority_set = set(CLINICAL_CLASS_INDICES)
        is_priority = torch.zeros(len(CLASS_NAMES), dtype=torch.bool)
        for idx in priority_set: is_priority[idx] = True
        self.register_buffer('is_priority', is_priority)

    def forward(self, loss_map: torch.Tensor, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        target_soft = batch_data['soft_label']
        target_hard = target_soft.argmax(dim=1)

        is_priority_device = self.is_priority.to(target_hard.device)
        priority_mask = is_priority_device[target_hard].float()

        other_mask = 1.0 - priority_mask

        priority_loss = (loss_map * priority_mask).sum() / (priority_mask.sum().clamp(min=1))
        other_loss = (loss_map * other_mask).sum() / (other_mask.sum().clamp(min=1))

        return self.priority_weight * priority_loss + (1.0 - self.priority_weight) * other_loss

class ComposableLoss(nn.Module):
    def __init__(self, base_loss_fn: nn.Module, weighters: List[nn.Module], reducer: nn.Module):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.weighters = nn.ModuleList(weighters)
        self.reducer = reducer

    def forward(self, logits: torch.Tensor, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.base_loss_fn.__class__.__name__ == 'WilliamsIndexLoss':
            return self.base_loss_fn(logits, batch_data)

        soft_label = batch_data['soft_label']
        pixel_loss = self.base_loss_fn(logits, soft_label)

        if self.weighters:
            final_weights = torch.ones_like(pixel_loss)
            for weighter in self.weighters:
                final_weights *= weighter(batch_data)
            pixel_loss *= final_weights

        return self.reducer(pixel_loss, batch_data)

def create_composable_segmentation_loss(loss_config: Dict, class_weights: torch.Tensor) -> nn.Module:
    if not isinstance(loss_config, dict):
        raise TypeError(f"loss_config must be dict, got {type(loss_config)}: {repr(loss_config)[:200]}")
    if 'name' not in loss_config:
        raise KeyError("loss_config missing 'name'")
    params = {**loss_config, 'class_weights': class_weights}
    base_loss_fn = LOSSES[params['name']](**params)

    weighters = []
    for scheme in params.get('weighting_schemes', []):
        if scheme['type'] == 'disagreement':
            weighters.append(DisagreementWeighter(scheme['metric'], scheme.get('epsilon', 1e-6)))
        elif scheme['type'] == 'scaled':
            weighters.append(ScaledWeighter(params['class_weights'], scheme['priority_scale_factor']))

    reducer_config = params.get('reducer', {'type': 'mean'})
    if reducer_config['type'] == 'priority_split':
        reducer = PrioritySplitReducer(reducer_config['priority_weight'])
    else:
        reducer = MeanReducer()

    return ComposableLoss(base_loss_fn, weighters, reducer)

class CombinedLoss(nn.Module):
    def __init__(self, main: Dict, aux: Dict, aux_weight: float = 0.4, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        if not isinstance(main, dict):
            raise TypeError(f"'main' loss config must be dict, got {type(main)}: {repr(main)[:200]}")
        if aux_weight > 1e-6 and not isinstance(aux, dict):
            raise TypeError(f"'aux' loss config must be dict when aux_weight>0, got {type(aux)}: {repr(aux)[:200]}")
        self.aux_weight = aux_weight
        self.main_loss_fn = create_composable_segmentation_loss(main, class_weights)

        self.aux_loss_fn = None
        if self.aux_weight > 0 and class_weights is not None:
            aux_params = {**aux, 'class_weights': class_weights}

            if any(s['type'] == 'scaled' for s in aux_params.get('weighting_schemes', [])):
                 weights = aux_params['class_weights'].clone()
                 scale_factor = next(s['priority_scale_factor'] for s in aux_params['weighting_schemes'] if s['type'] == 'scaled')
                 for idx in CLINICAL_CLASS_INDICES: weights[idx] *= scale_factor
                 aux_params['class_weights'] = weights

            if aux_params['name'] in LOSSES:
                self.aux_loss_fn = LOSSES[aux_params['name']](**aux_params)

    def forward(self, outputs: Tuple[torch.Tensor, Optional[torch.Tensor]], batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        main_out, aux_out = outputs
        main_loss = self.main_loss_fn(main_out, batch_data)

        if aux_out is None or self.aux_weight == 0 or self.aux_loss_fn is None:
            return main_loss

        num_classes = main_out.shape[1]
        soft_label = batch_data['soft_label']
        target_hard_seg = soft_label.argmax(dim=1)

        target_flat = target_hard_seg.view(target_hard_seg.shape[0], -1)
        counts = F.one_hot(target_flat, num_classes).float().sum(dim=1)

        counts[:, 0] = 0
        aux_targets = counts.argmax(dim=1)

        aux_loss = self.aux_loss_fn(aux_out, aux_targets).mean()

        return main_loss + self.aux_weight * aux_loss