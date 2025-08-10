import h5py
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple, Union, Set
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

try:
    import optuna
    TRIAL_PRUNED_EXCEPTION = optuna.exceptions.TrialPruned
except ImportError:
    class TrialPruned(Exception): pass
    TRIAL_PRUNED_EXCEPTION = TrialPruned

from .pspnet import KorniaTrainAugmentation
from .losses import CombinedLoss
from .evaluator import METRIC_REGISTRY, _required_batch_keys

class GleasonH5Dataset(Dataset):
    def __init__(self, h5_path: Path, split: str, required_keys: Optional[Set[str]] = None):
        self.h5_path = str(h5_path)
        self.split = split
        self._file: Optional[h5py.File] = None

        self.dsets: Dict[str, h5py.Dataset] = {}
        self._len: Optional[int] = None

        self.keys_to_load = {'images', 'soft_labels'}
        if required_keys:
            self.keys_to_load.update(required_keys)

    def reopen(self):
        if self._file is not None:
            self._file.close()
        self._file = h5py.File(self.h5_path, 'r', swmr=True, libver='latest')
        split_group = self._file[self.split]
        self.dsets = {k: split_group[k] for k in self.keys_to_load if k in split_group}

    def _ensure_open(self):
        if self._file is None:
            self.reopen()

    def __len__(self) -> int:
        if self._len is None:
             with h5py.File(self.h5_path, 'r') as f:
                self._len = len(f[self.split]['images'])
        return self._len

    def __getitem__(self, idx: int) -> Dict:
        self._ensure_open()

        item = {name: torch.from_numpy(dset[idx]) for name, dset in self.dsets.items()}
        if 'images' in item: item['image'] = item.pop('images')
        if 'soft_labels' in item: item['soft_label'] = item.pop('soft_labels')
        return item

    def __del__(self):
        if self._file:
            self._file.close()

class Trainer:
    def __init__(self, model: nn.Module, device: torch.device, config: Dict, use_amp: bool = True, is_hpo_run: bool = False):
        self.device  = device
        self.config  = config
        self.use_amp = use_amp and torch.cuda.is_available()
        self.is_hpo_run = is_hpo_run

        self.model = model
        self.train_augmenter = KorniaTrainAugmentation(config.get('img_size', 512)).to(device)
        self.grad_accum_steps = config.get('gradient_accumulation_steps', 1)
        self.criterion = CombinedLoss(**config.get('loss', {})).to(device)
        self.early_stopping_patience = config.get('early_stopping_patience', 10)

        self.objective_metric = config.get('objective_metric', 'loss')
        scheduler_mode = 'min' if 'loss' in self.objective_metric else 'max'

        self._configure_metrics()

        self.optimiser = torch.optim.AdamW(
            self.model.parameters(), lr=config.get('learning_rate', 1e-4),
            betas=(0.9, 0.999), weight_decay=config.get('weight_decay', 0.01),
            fused=self.use_amp
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser, mode=scheduler_mode, factor=0.5, patience=config.get('scheduler_patience', 5)
        )

        self.scaler            = torch.amp.GradScaler(enabled=self.use_amp)
        self.history           = defaultdict(list)
        self.best_score        = float('-inf') if scheduler_mode == 'max' else float('inf')
        self.best_metrics      = {}
        self.epochs_no_improve = 0
        self.cuda_stream       = torch.cuda.Stream() if self.use_amp else None

    def _configure_metrics(self):
        self.val_metrics_to_run: Dict[str, callable] = {**METRIC_REGISTRY}

        self.required_val_keys: Set[str] = set()
        for func in self.val_metrics_to_run.values():
            self.required_val_keys.update(_required_batch_keys(func))

        self.val_needs_pred_probs = 'pred_probs' in self.required_val_keys

        self.required_dataset_keys = self.required_val_keys - {'pred_hard', 'pred_probs', 'target_hard'}

    def train_epoch(self, dataloader: DataLoader, epoch: int, total_epochs: int) -> Dict[str, float]:
        self.model.train()
        total_loss, num_batches = 0.0, len(dataloader)
        alpha = min(1.0, epoch / (total_epochs * 0.75))

        iterable = dataloader
        if not self.is_hpo_run:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
            iterable = pbar

        for i, batch_data in enumerate(iterable):
            with torch.cuda.stream(self.cuda_stream):
                batch_gpu = {k: v.to(self.device, non_blocking=True) for k, v in batch_data.items()}
                images_raw = batch_gpu['image']
                
                images_aug = self.train_augmenter(images_raw, alpha=alpha)
                
                loss_batch_data = {**batch_gpu, 'image': images_aug}

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(images_aug, aux_weight=self.criterion.aux_weight)
                    loss = self.criterion(outputs, loss_batch_data)
                    unscaled_loss = loss.item()
                    loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == num_batches:
                self.scaler.unscale_(self.optimiser)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimiser)
                self.scaler.update()
                self.optimiser.zero_grad(set_to_none=True)

            total_loss  += unscaled_loss
            if not self.is_hpo_run:
                pbar.set_postfix(train_loss=f"{total_loss / (i + 1):.4f}")

        torch.cuda.synchronize()
        return {'loss': total_loss / num_batches}

    @torch.inference_mode()
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        batch_metrics = defaultdict(list)
        total_loss, num_samples = 0.0, 0

        for batch_data in dataloader:
            batch_gpu = {k: v.to(self.device, non_blocking=True) for k, v in batch_data.items()}
            images = batch_gpu['image']
            batch_size = images.shape[0]

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(images, aux_weight=self.criterion.aux_weight)
                loss = self.criterion(outputs, batch_gpu)

            total_loss += loss.item() * batch_size
            num_samples += batch_size

            metric_input_data = {**batch_gpu}
            metric_input_data['pred_hard'] = outputs[0].argmax(dim=1)
            metric_input_data['target_hard'] = batch_gpu['soft_label'].argmax(dim=1)
            if self.val_needs_pred_probs:
                metric_input_data['pred_probs'] = F.softmax(outputs[0], dim=1)

            for name, func in self.val_metrics_to_run.items():
                required = _required_batch_keys(func)
                if all(k in metric_input_data for k in required):
                    kwargs = {k: metric_input_data[k] for k in required}
                    metric_val = func(self, **kwargs)
                    if metric_val is not None and metric_val.numel() > 0:
                        batch_metrics[name].append(metric_val.cpu())

        agg_metrics = {'loss': total_loss / num_samples if num_samples > 0 else 0.0}
        for name, values in batch_metrics.items():
            if values:
                agg_metrics[name] = torch.cat(values).mean().item()

        return agg_metrics

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader],
            num_epochs: int, save_dir: Path, verbose: bool = True,
            trial: Optional[any] = None, trial_num: int = 0):
        save_model      = self.config.get('save_model_weights', True)
        best_model_path = save_dir / 'best_model.pth'
        if save_model: save_dir.mkdir(parents=True, exist_ok=True)

        if verbose and not trial:
            print(f"Starting Training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch, num_epochs)
            self.history['train_loss'].append(train_metrics['loss'])

            if val_loader:
                val_metrics = self.validate_epoch(val_loader)
                for name, value in val_metrics.items():
                    self.history[f'val_{name}'].append(value)
            else:
                val_metrics = {'loss': train_metrics['loss']}

            current_score = val_metrics.get(self.objective_metric, -val_metrics.get('loss', 0.0))

            if verbose and not trial:
                metric_str = " | ".join([f"Val {k.replace('_', ' ').title()}: {v:.4f}" for k,v in val_metrics.items()])
                print(f"Epoch {epoch}/{num_epochs} - {metric_str}")

            self.scheduler.step(current_score)

            is_better = (self.scheduler.mode == 'max' and current_score > self.best_score) or \
                        (self.scheduler.mode == 'min' and current_score < self.best_score)

            if is_better:
                self.best_score, self.epochs_no_improve = current_score, 0
                self.history['best_epoch'] = epoch
                self.best_metrics = val_metrics
                if save_model: self._save_checkpoint(best_model_path)
            else:
                self.epochs_no_improve += 1

            if trial:
                trial.report(current_score, epoch)
                if trial.should_prune(): raise TRIAL_PRUNED_EXCEPTION()

            if self.epochs_no_improve >= self.early_stopping_patience:
                if verbose: print(f"Stopping early after {epoch} epochs.")
                break

        if save_model and best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            uncompiled_model = self.model.net._orig_mod if hasattr(self.model.net, '_orig_mod') else self.model.net
            uncompiled_model.load_state_dict(checkpoint['model_state_dict'])

        return {
            'history': dict(self.history), 'best_score': self.best_score,
            'best_epoch': self.history.get('best_epoch', -1),
            'best_metrics': self.best_metrics
        }

    def _save_checkpoint(self, path: Path):
        uncompiled_net = self.model.net._orig_mod if hasattr(self.model.net, '_orig_mod') else self.model.net
        torch.save({'model_state_dict': uncompiled_net.state_dict(), 'config': self.config}, path)

def _worker_init_reopen(worker_id: int):
    wi = torch.utils.data.get_worker_info()
    if wi is None:
        return
    ds = wi.dataset
    if hasattr(ds, "reopen"):
        ds.reopen()

def create_data_loaders(h5_path, batch_size, num_workers, img_size, pin_memory, config=None, required_val_keys=None):
    required_train_keys = set()
    if config:
        loss_cfg = config.get('loss', {})
        main_cfg = loss_cfg.get('main', {})
        if main_cfg.get('name') == 'williams_index':
            required_train_keys.add('individual_masks')
        for scheme in main_cfg.get('weighting_schemes', []):
            if scheme['type'] == 'disagreement':
                required_train_keys.add(f"disagreement_map_{scheme['metric']}")

    train_ds = GleasonH5Dataset(h5_path=h5_path, split="train", required_keys=required_train_keys)
    val_ds   = GleasonH5Dataset(h5_path=h5_path, split="val", required_keys=required_val_keys)

    train_sampler = None
    with h5py.File(h5_path, 'r') as f:
        if 'train' in f and 'sample_weights' in f['train']:
            sample_weights = torch.from_numpy(f['train/sample_weights'][:])
            train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    common = dict(
        num_workers=max(2, int(num_workers)),
        worker_init_fn=_worker_init_reopen,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=bool(pin_memory),
        pin_memory_device=("cuda" if torch.cuda.is_available() else ""),
        multiprocessing_context="spawn",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader