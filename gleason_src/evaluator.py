import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from collections import defaultdict
import h5py
import json
import pickle
from itertools import combinations
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import inspect

from gleason_src.preprocessing import CLINICAL_CLASS_INDICES, CLASS_NAMES
from gleason_src.models_utils import build_model, ensure_num_classes

METRIC_REGISTRY: Dict[str, callable] = {}

def register_metric(name: str):
    def decorator(func):
        METRIC_REGISTRY[name] = func
        return func
    return decorator

def _required_batch_keys(fn: callable) -> Tuple[str, ...]:
    sig = inspect.signature(fn)
    req = []
    for p in sig.parameters.values():
        if p.name == "self" or p.name == "kwargs":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p.default is inspect.Parameter.empty:
            req.append(p.name)
    return tuple(req)

def _calculate_gleason_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    gleason_scores = []
    target = target.to(pred.device)
    for c in CLINICAL_CLASS_INDICES:
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum(dim=(-2, -1))
        union = pred_c.sum(dim=(-2, -1)) + target_c.sum(dim=(-2, -1))
        dice = (2 * intersection + smooth) / (union + smooth)
        gleason_scores.append(dice)
    return torch.stack(gleason_scores, dim=1).mean(dim=1)

class ModelEvaluator:
    def __init__(self, model: nn.Module, device: torch.device, config: Dict):
        self.device = device
        self.config = config

        if 'num_classes' not in self.config and hasattr(model, 'num_classes'):
            self.config['num_classes'] = model.num_classes

        self.model = model.to(self.device).eval()
        self.batch_size = config.get('batch_size', 32)
        self.use_amp = torch.cuda.is_available()

        include = self.config.get('metrics', None)
        exclude = set(self.config.get('metrics_exclude', []))

        if include is None or include == 'all':
            chosen_names = [n for n in METRIC_REGISTRY.keys() if n not in exclude]
        else:
            chosen_names = [n for n in include if n in METRIC_REGISTRY and n not in exclude]

        self.metric_functions: Dict[str, callable] = {n: METRIC_REGISTRY[n] for n in chosen_names}

        self.metric_requirements: Dict[str, Tuple[str, ...]] = {
            name: _required_batch_keys(fn) for name, fn in self.metric_functions.items()
        }
        self._needs_pred_probs = any('pred_probs' in req for req in self.metric_requirements.values())

    @torch.inference_mode()
    def evaluate_comprehensive(self, test_dataset: Dataset, save_dir: Optional[Path] = None) -> Dict:
        self.model.eval()
        eval_batch_size = self.batch_size * 2

        nw = max(2, int(getattr(self, "num_workers", 4)))
        common = dict(
            num_workers=nw,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=torch.cuda.is_available(),
            pin_memory_device=("cuda" if torch.cuda.is_available() else ""),
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        )

        results = {'metrics': defaultdict(list), 'raw': {}}
        all_preds_flat, all_targets_flat = [], []
        model_pathologist_dice = defaultdict(list)
        inter_pathologist_dice = defaultdict(list)

        if hasattr(test_dataset, '_ensure_open'):
            test_dataset._ensure_open()

        pathologists = []
        if hasattr(test_dataset, 'dsets') and test_dataset.dsets:
            any_dset_key = next(iter(test_dataset.dsets.keys()))
            pathologists_json = test_dataset.dsets[any_dset_key].file.attrs.get('pathologists', '[]')
            if pathologists_json:
                pathologists = json.loads(pathologists_json)

        pbar = tqdm(test_loader, desc="Evaluating", unit="batch", leave=False)
        for batch in pbar:
            batch_gpu = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            images = batch_gpu['image']

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                if self.config.get('perform_mc_dropout', False):
                    n_mc_samples = self.config.get('mc_samples', 10)
                    with torch.backends.cudnn.flags(benchmark=False):
                        uncompiled_net = self.model.net._orig_mod if hasattr(self.model.net, '_orig_mod') else self.model.net

                        images_mc = images
                        if images_mc.ndim == 4 and images_mc.shape[-1] == 3:
                            images_mc = images_mc.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
                        if images_mc.shape[-2:] != (self.model.img_size, self.model.img_size):
                            images_mc = F.interpolate(images_mc, size=(self.model.img_size, self.model.img_size), mode="bilinear", align_corners=False)
                        images_mc = self.model.norm(images_mc)

                        final_out, uncertainty_map = uncompiled_net.monte_carlo_predict(images_mc, n_samples=n_mc_samples)

                    avg_uncertainty = uncertainty_map.mean(dim=(1, 2)).detach().cpu().tolist()
                    results['raw']['uncertainty'] = results['raw'].get('uncertainty', []) + avg_uncertainty
                else:
                    final_out = self.model(images)

            pred_hard = final_out.argmax(dim=1)
            target_hard = batch_gpu['soft_label'].argmax(dim=1)

            batch_data = {
                'pred_hard': pred_hard,
                'target_hard': target_hard,
                **batch_gpu
            }

            if self._needs_pred_probs:
                if self.config.get('perform_mc_dropout', False):
                    batch_data['pred_probs'] = final_out
                else:
                    batch_data['pred_probs'] = F.softmax(final_out, dim=1)

            for name, func in self.metric_functions.items():
                req = self.metric_requirements[name]
                if any(r not in batch_data for r in req):
                    continue
                
                kwargs_for_func = {key: batch_data[key] for key in req if key in batch_data}
                metric_val = func(self, **kwargs_for_func)

                if metric_val is not None:
                    results['metrics'][name].append(metric_val.detach().cpu())

            tissue_mask = target_hard > 0
            if tissue_mask.any():
                all_preds_flat.append(pred_hard[tissue_mask].detach().cpu().numpy().flatten())
                all_targets_flat.append(target_hard[tissue_mask].detach().cpu().numpy().flatten())
            
            individual_masks = batch_gpu.get('individual_masks')
            has_individual_masks = individual_masks is not None and individual_masks.nelement() > 0
            if has_individual_masks:
                for r_idx, r_name in enumerate(pathologists):
                    expert_mask = individual_masks[:, r_idx]
                    model_pathologist_dice[r_name].extend(
                        _calculate_gleason_dice(pred_hard, expert_mask).detach().cpu().tolist()
                    )

                for p1_idx, p2_idx in combinations(range(len(pathologists)), 2):
                    p1_masks, p2_masks = individual_masks[:, p1_idx], individual_masks[:, p2_idx]
                    pair_name = f"{pathologists[p1_idx].replace('_T','')}-{pathologists[p2_idx].replace('_T','')}"
                    inter_pathologist_dice[pair_name].extend(
                        _calculate_gleason_dice(p1_masks, p2_masks).detach().cpu().tolist()
                    )

        for key, val_list in results['metrics'].items():
            if val_list:
                results['raw'][key] = torch.cat(val_list, dim=0).numpy() if isinstance(val_list[0], torch.Tensor) else np.array(val_list)

        preds_flat = np.concatenate(all_preds_flat) if all_preds_flat else np.array([])
        targets_flat = np.concatenate(all_targets_flat) if all_targets_flat else np.array([])
        summary = self._aggregate_results(results, model_pathologist_dice, inter_pathologist_dice, preds_flat, targets_flat)

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=float)
            with open(save_dir / 'detailed_results.pkl', 'wb') as f:
                pickle.dump({'summary': summary, 'raw': results['raw']}, f)

        return {'detailed_results': results['raw'], 'summary': summary}

    def _aggregate_results(self, results, model_dice, inter_dice, all_preds, all_targets):
        summary = {}
        for metric, values in results['raw'].items():
            if not isinstance(values, np.ndarray) or values.size == 0:
                continue
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
            if values.ndim > 1:
                summary[f'{metric}_per_class_mean'] = np.mean(values, axis=0).tolist()
                summary[f'{metric}_per_class_std'] = np.std(values, axis=0).tolist()

        summary['model_pathologist_gleason_dice'] = {p: np.mean(s) for p, s in model_dice.items() if s}
        summary['inter_pathologist_gleason_dice'] = {p: np.mean(s) for p, s in inter_dice.items() if s}

        summary['class_names'] = CLASS_NAMES[1:]
        if all_preds.size > 0 and all_targets.size > 0:
            cm = confusion_matrix(all_targets, all_preds, labels=range(1, self.config['num_classes']))
            cm_sum = cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.divide(cm.astype('float'), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0)
            summary['confusion_matrix'] = cm_norm.tolist()
        else:
            summary['confusion_matrix'] = []

        return summary

@register_metric('pixel_accuracy')
def compute_pixel_accuracy(self, pred_hard: torch.Tensor, target_hard: torch.Tensor) -> torch.Tensor:
    tissue_mask = target_hard > 0
    if not tissue_mask.any():
        return torch.tensor([], device=pred_hard.device)

    accuracies = []
    for i in range(pred_hard.shape[0]):
        sample_mask = tissue_mask[i]
        if sample_mask.any():
            acc = (pred_hard[i][sample_mask] == target_hard[i][sample_mask]).float().mean()
            accuracies.append(acc)

    return torch.tensor(accuracies, device=pred_hard.device) if accuracies else torch.tensor([], device=pred_hard.device)

@register_metric('dice_score')
def batch_dice_score(self, pred_hard: torch.Tensor, target_hard: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    dices = []
    dims = (1, 2)
    for c in CLINICAL_CLASS_INDICES:
        pred_c = (pred_hard == c)
        target_c = (target_hard == c)
        intersection = torch.sum(pred_c & target_c, dim=dims).float()
        union = torch.sum(pred_c, dim=dims).float() + torch.sum(target_c, dim=dims).float()
        dice = (2. * intersection + smooth) / (union + smooth)
        dices.append(dice)
    return torch.stack(dices, dim=1)

@register_metric('iou_score')
def batch_iou_score(self, pred_hard: torch.Tensor, target_hard: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    ious = []
    dims = (1, 2)
    for c in CLINICAL_CLASS_INDICES:
        pred_c = (pred_hard == c)
        target_c = (target_hard == c)
        intersection = torch.sum(pred_c & target_c, dim=dims).float()
        union = torch.sum(pred_c | target_c, dim=dims).float()
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)
    return torch.stack(ious, dim=1)

@register_metric('gleason_dice_score')
def batch_gleason_dice_score(self, pred_hard: torch.Tensor, target_hard: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    return _calculate_gleason_dice(pred_hard, target_hard, smooth)

@register_metric('williams_index')
def batch_williams_index(self, pred_hard: torch.Tensor, individual_masks: torch.Tensor, inter_expert_error_map: torch.Tensor) -> Optional[torch.Tensor]:
    if individual_masks.shape[1] < 2:
        return None
    epsilon = 1e-10

    tissue_mask = torch.any(individual_masks > 0, dim=1, keepdim=True)
    if not tissue_mask.any():
        return torch.tensor([], device=pred_hard.device)

    pred_expanded = pred_hard.unsqueeze(1)
    model_expert_diff = (pred_expanded != individual_masks).float()
    
    model_expert_error_per_sample = (model_expert_diff * tissue_mask).sum(dim=(2, 3)) / tissue_mask.sum(dim=(2, 3)).clamp(min=1)
    avg_model_expert_error = model_expert_error_per_sample.mean(dim=1)
    
    avg_expert_expert_error = (inter_expert_error_map.squeeze(1) * tissue_mask.squeeze(1)).sum(dim=(1, 2)) / tissue_mask.squeeze(1).sum(dim=(1, 2)).clamp(min=1)

    inverse_model_agreement = 1.0 / (avg_model_expert_error + epsilon)
    inverse_expert_agreement = 1.0 / (avg_expert_expert_error + epsilon)

    williams_index = inverse_model_agreement / (inverse_expert_agreement + epsilon)
    
    return williams_index

@register_metric('grade_aware_williams_index')
def batch_grade_aware_williams_index(self, pred_probs: torch.Tensor, individual_masks: torch.Tensor, inter_expert_emd_map: torch.Tensor) -> Optional[torch.Tensor]:
    if individual_masks.shape[1] < 2:
        return None
    epsilon = 1e-10
    C = self.config['num_classes']

    tissue_mask = torch.any(individual_masks > 0, dim=1, keepdim=True)
    if not tissue_mask.any():
        return torch.tensor([], device=pred_probs.device)

    expert_masks_one_hot = F.one_hot(individual_masks.long(), num_classes=C).permute(0, 1, 4, 2, 3).float()

    model_cdf = torch.cumsum(pred_probs.unsqueeze(1), dim=2)
    expert_cdf = torch.cumsum(expert_masks_one_hot, dim=2)

    expert_model_emd_map = torch.abs(model_cdf - expert_cdf).sum(dim=2)
    expert_model_emd_per_sample = (expert_model_emd_map * tissue_mask).sum(dim=(2, 3)) / tissue_mask.sum(dim=(2, 3)).clamp(min=1)
    avg_model_expert_emd = expert_model_emd_per_sample.mean(dim=1)

    avg_expert_expert_emd = (inter_expert_emd_map.squeeze(1) * tissue_mask.squeeze(1)).sum(dim=(1,2)) / tissue_mask.squeeze(1).sum(dim=(1,2)).clamp(min=1)

    inverse_model_agreement = 1.0 / (avg_model_expert_emd + epsilon)
    inverse_expert_agreement = 1.0 / (avg_expert_expert_emd + epsilon)

    gawi = inverse_model_agreement / (inverse_expert_agreement + epsilon)

    return gawi

@register_metric('kappa_f1_combo')
def kappa_f1_combo(self,
                   pred_hard: torch.Tensor,
                   target_hard: torch.Tensor) -> torch.Tensor:
    device = pred_hard.device
    B = pred_hard.size(0)
    C = self.config['num_classes']

    results = []
    for i in range(B):
        pred_sample = pred_hard[i]
        targ_sample = target_hard[i]

        tissue_mask = targ_sample > 0
        if not tissue_mask.any():
            continue

        pred = pred_sample[tissue_mask]
        targ = targ_sample[tissue_mask]

        cm = torch.bincount(targ * C + pred, minlength=C*C).reshape(C, C).float()

        tp = cm.diag()
        rowsum = cm.sum(dim=1)
        colsum = cm.sum(dim=0)
        N = cm.sum()

        po = tp.sum() / (N + 1e-12)
        pe = (rowsum * colsum).sum() / (N * N + 1e-12)
        kappa = (po - pe) / (1.0 - pe + 1e-12)

        fp = colsum - tp
        fn = rowsum - tp
        denom = (2 * tp + fp + fn)
        macro_f1_per_class = torch.where(denom > 0, (2 * tp) / (denom + 1e-12), torch.zeros_like(denom))
        macro_f1 = macro_f1_per_class.mean()
        micro_f1 = po

        results.append(kappa + 0.5 * (macro_f1 + micro_f1))

    return torch.tensor(results, device=device) if results else torch.tensor([], device=device)

def evaluate_model(model_path: Path, test_dataset: Dataset, device: torch.device, config: Dict, save_dir: Optional[Path] = None) -> Dict:
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception:
        checkpoint = torch.load(model_path, map_location=device)

    model_config = checkpoint.get('config', {})
    final_config = ensure_num_classes({**model_config, **config})

    model = build_model(
        backbone=final_config.get("backbone", "resnet152"),
        dropout=final_config.get("dropout_rate", 0.1),
        img_size=final_config.get("img_size", 512),
        num_classes=final_config.get("num_classes", 5),
        mode="compiled"
    )

    uncompiled_net = model.net._orig_mod if hasattr(model.net, '_orig_mod') else model.net
    uncompiled_net.load_state_dict(checkpoint['model_state_dict'])

    evaluator = ModelEvaluator(model, device, final_config)
    return evaluator.evaluate_comprehensive(test_dataset, save_dir)