import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import h5py
import torch
import os, subprocess, torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocessing import (
    scan_directory, compute_agreement_matrices_vectorised,
    compute_reliability_weights, save_dataset_hdf5, split_data_stratified,
    compute_class_weights, compute_sample_weights, CLASS_NAMES
)
from .visualisations import (
    plot_pathologist_agreement_by_class, plot_pathologist_reliability,
    plot_evaluation_summary, plot_confusion_matrix, plot_training_curves
)
from .pspnet import PSPNet
from .evaluator import evaluate_model
from .trainer import Trainer, create_data_loaders
from .models_utils import build_model, ensure_num_classes

def _set_speed_flags():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

def run_preprocessing_pipeline(
    source_folder: Path,
    output_dir: Path,
    pathologists: Optional[List[str]] = None,
    target_size: Tuple[int, int] = (512, 512),
    strict_filtering: bool = True,
    force_rerun: bool = False
) -> Dict:
    pathologists = pathologists or ['Maps1_T', 'Maps3_T', 'Maps4_T', 'Maps5_T']
    datasets_dir = output_dir / 'datasets'
    metadata_path = output_dir / 'metadata.json'
    h5_path = datasets_dir / 'gleason_dataset.h5'

    if not force_rerun and h5_path.exists() and metadata_path.exists():
        print("Preprocessing outputs found. Skipping.")
        with open(metadata_path, 'r') as f: return json.load(f)

    if h5_path.exists():
        h5_path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60, "\nGLEASON GRADING PREPROCESSING PIPELINE\n" + "=" * 60)

    registry = scan_directory(Path(source_folder), pathologists)
    train_df, val_df, test_df = split_data_stratified(registry, pathologists, strict=strict_filtering)

    agreement_matrices = compute_agreement_matrices_vectorised(registry, pathologists, target_size)
    pathologist_weights = compute_reliability_weights(agreement_matrices, pathologists)

    datasets_dir.mkdir(exist_ok=True)
    save_dataset_hdf5(train_df.to_dict('records'), pathologist_weights, h5_path, pathologists, target_size, split_name='train')
    save_dataset_hdf5(val_df.to_dict('records'), pathologist_weights, h5_path, pathologists, target_size, split_name='val')
    save_dataset_hdf5(test_df.to_dict('records'), pathologist_weights, h5_path, pathologists, target_size, split_name='test')

    if h5_path.exists() and len(train_df) > 0:
        class_w = compute_class_weights(h5_path, split_name='train')
        sample_w = compute_sample_weights(h5_path, class_w, split_name='train')
        with h5py.File(h5_path, 'a') as f:
            train_group = f['train']
            if 'class_weights' not in train_group:
                train_group.create_dataset('class_weights', data=class_w)
            if 'sample_weights' not in train_group:
                train_group.create_dataset('sample_weights', data=sample_w)
        print(f"{'Computed class & sample weights':<35}: Done")

    metadata = {
        'pathologists': pathologists,
        'target_size': target_size,
        'split_sizes': {'train': len(train_df), 'val': len(val_df), 'test': len(test_df)},
        'agreement_matrices': {k: v.tolist() for k,v in agreement_matrices.items()},
        'pathologist_weights': pathologist_weights
    }
    with open(metadata_path, 'w') as f: json.dump(metadata, f, indent=2)

    vis_dir = output_dir / 'visualisations'
    plot_pathologist_agreement_by_class(metadata_path, save_path=vis_dir / 'pathologist_agreement.png')
    plot_pathologist_reliability(metadata_path, save_path=vis_dir / 'pathologist_reliability.png')

    print(f"\nPreprocessing complete! Outputs saved to: {output_dir}")
    return metadata

def run_training_pipeline(datasets_dir: Path, output_dir: Path, experiment_name: str, config: Dict, verbose: bool = True, trial: Optional[any] = None, trial_num: int = 0, is_hpo_run: bool = False) -> Dict:
    h5_path    = datasets_dir / 'gleason_dataset.h5'
    device     = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    num_epochs = config.get('num_epochs', 25)

    config = ensure_num_classes(config)

    model = build_model(
        backbone=config.get("backbone", "resnet152"),
        dropout=config.get("dropout_rate", 0.1),
        img_size=config.get("img_size", 512),
        num_classes=config.get("num_classes", 5),
        mode="uncompiled"
    )

    trainer = Trainer(model, device, config, is_hpo_run=is_hpo_run)

    train_loader, val_loader = create_data_loaders(
        h5_path=h5_path,
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 0),
        img_size=config.get('img_size', 512),
        pin_memory=(device.type == 'cuda'),
        config=config,
        required_val_keys=trainer.required_dataset_keys
    )

    model_dir = output_dir / 'models' / experiment_name
    results   = trainer.fit(train_loader, val_loader, num_epochs, model_dir,
                            verbose=verbose, trial=trial, trial_num=trial_num)

    if verbose and not trial:
        plot_training_curves(results['history'], save_path=model_dir / 'training_curves.png')
        with open(model_dir / 'config.json', 'w') as f: json.dump(trainer.config, f, indent=2, default=lambda o: '<not serialisable>')
        print(f"\nTraining complete! Model info saved to: {model_dir}")
    return results

def run_evaluation_pipeline(model_path: Path, test_data_path: Path, output_dir: Path, config: Dict, verbose: bool = True) -> Dict:
    from .trainer import GleasonH5Dataset

    device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if verbose: print(f"Evaluating model: {model_path} on {device}")

    eval_dir = output_dir / 'evaluation'

    test_dataset = GleasonH5Dataset(h5_path=test_data_path, split='test', required_keys={'individual_masks', 'inter_expert_error_map', 'inter_expert_emd_map'})
    results = evaluate_model(model_path, test_dataset, device, config, save_dir=eval_dir)

    if 'summary' in results and verbose:
        summary_path = eval_dir / 'summary.json'
        primary_metric = config.get('objective_metric', 'williams_index')
        plot_evaluation_summary(summary_path, save_path=eval_dir / 'summary.png', primary_metric=primary_metric)
        plot_confusion_matrix(summary_path, save_path=eval_dir / 'confusion_matrix.png')

    if verbose: print(f"\nEvaluation complete! Results saved to: {eval_dir}")
    return results