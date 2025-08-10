import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import seaborn as sns
import json
import pickle
import cv2
import h5py
import torch
import torch.nn.functional as F
import matplotlib.patches as mpatches
from tqdm import tqdm

from .preprocessing import CLASS_NAMES
from .utils import save_or_show, _format_pathologist_name
from .pspnet import PSPNet, KorniaTrainAugmentation, KorniaValidationAugmentation
from .models_utils import build_model

def plot_training_curves(history: Dict[str, List[float]], save_path: Optional[Path] = None, display_in_notebook: bool = False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Model Learning Curves', fontsize=16)

    epochs = np.arange(1, len(history.get('train_loss', [])) + 1)
    if not epochs.any(): return

    axes[0].plot(epochs, history.get('train_loss', []), 'b-', label='Train Loss')
    if 'val_loss' in history:
        val_epochs = np.arange(1, len(history['val_loss']) + 1)
        axes[0].plot(val_epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set(xlabel='Epoch', ylabel='Loss', title='Training & Validation Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    score_key = next((k for k in history.keys() if 'val_' in k and k != 'val_loss' and 'epoch' not in k), None)
    if score_key:
        score_name = score_key.replace('val_', '').replace('_', ' ').title()
        score_epochs_key = f"{score_key}_epoch"
        score_epochs = history.get(score_epochs_key, epochs)
        score_values = history.get(score_key, [])
        if score_values:
            axes[1].plot(score_epochs, score_values, 'g-', marker='o', linestyle='--', label=f'Val {score_name}')
            axes[1].set(xlabel='Epoch', ylabel=score_name, title='Validation Score')
            axes[1].legend(); axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No validation score available.", ha='center', va='center', style='italic', transform=axes[1].transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.96]); save_or_show(fig, save_path, display_in_notebook)

def plot_pathologist_agreement_by_class(metadata_path: Path, save_path: Optional[Path] = None, display_in_notebook: bool = False):
    if not metadata_path.exists(): return
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    agreement_matrices = metadata.get('agreement_matrices', {})
    p_names = metadata.get('pathologists', [])
    if not agreement_matrices or not p_names: return

    class_items = sorted(agreement_matrices.items())
    if not class_items: return
    cols = min(3, len(class_items)); rows = (len(class_items) + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 5 + 1.5, rows * 5))
    gs = gridspec.GridSpec(rows, cols + 1, width_ratios=[5] * cols + [0.5])
    cbar_ax = fig.add_subplot(gs[:, -1])

    for i, (class_name, matrix_list) in enumerate(class_items):
        matrix = np.array(matrix_list)
        ax = fig.add_subplot(gs[i // cols, i % cols])
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.0, vmax=1.0, ax=ax, cbar=i==0, cbar_ax=cbar_ax if i==0 else None)
        ax.set_title(class_name.replace('_', ' ')); ax.set_aspect('equal', 'box')
        ax.set_xticklabels([_format_pathologist_name(n) for n in p_names], rotation=45, ha="right")
        ax.set_yticklabels([_format_pathologist_name(n) for n in p_names], rotation=0)
    fig.suptitle('Pairwise Inter-Pathologist Agreement by Tissue Class', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95]); save_or_show(fig, save_path, display_in_notebook)

def plot_pathologist_reliability(metadata_path: Path, save_path: Optional[Path] = None, display_in_notebook: bool = False):
    if not metadata_path.exists(): return
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    pathologist_weights = metadata.get('pathologist_weights', {})
    if not pathologist_weights: return

    df = pd.DataFrame(pathologist_weights).T.reset_index().melt(id_vars='index', var_name='Class', value_name='Reliability')
    df['Pathologist'] = df['index'].apply(_format_pathologist_name)
    df['Class'] = df['Class'].str.replace('_', ' ')

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x='Class', y='Reliability', hue='Pathologist', ax=ax, palette='viridis')
    ax.set(ylabel='Reliability Score', title='Pathologist Reliability Score by Tissue Class')
    ax.legend(title='Pathologist', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right'); save_or_show(fig, save_path, display_in_notebook)

def plot_evaluation_summary(summary_path: Path, save_path: Optional[Path] = None, display_in_notebook: bool = False, primary_metric: str = 'williams_index'):
    if not summary_path.exists(): return
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    fig = plt.figure(figsize=(14, 10)); gs = fig.add_gridspec(2, 1, hspace=0.4)
    fig.suptitle('Comprehensive Model Evaluation Summary', fontsize=16)

    ax1 = fig.add_subplot(gs[0, 0])
    metric_mean_key = f'{primary_metric}_mean'
    metric_std_key = f'{primary_metric}_std'
    metric_mean = summary.get(metric_mean_key)
    metric_std = summary.get(metric_std_key)
    metric_title = primary_metric.replace('_', ' ').title()
    ax1.set_title(f'Primary Performance Metric: {metric_title}')

    if metric_mean is not None:
        sns.barplot(x=[metric_title], y=[metric_mean], ax=ax1, palette=['#2ecc71'])
        if metric_std is not None:
            ax1.errorbar(x=[0], y=[metric_mean], yerr=[metric_std], fmt='none', c='black', capsize=5)
        for p in ax1.patches: ax1.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
    else:
        ax1.text(0.5, 0.5, f"Metric '{primary_metric}' not available", ha='center', va='center', style='italic')

    ax2 = fig.add_subplot(gs[1, 0]); ax2.set_title('Confusion Matrix')
    cm = np.array(summary.get('confusion_matrix', []))
    class_names = summary.get('class_names', CLASS_NAMES)
    if cm.size > 0:
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax2, xticklabels=class_names, yticklabels=class_names)
        ax2.set(xlabel='Predicted Label', ylabel='True Label')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax2.get_yticklabels(), rotation=0)
    else:
        ax2.text(0.5, 0.5, 'Matrix not available', ha='center', va='center', style='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.95]); save_or_show(fig, save_path, display_in_notebook)

def plot_confusion_matrix(summary_path: Path, save_path: Optional[Path] = None, display_in_notebook: bool = False):
    if not summary_path.exists(): return
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    cm = np.array(summary.get('confusion_matrix', []))
    class_names = summary.get('class_names', CLASS_NAMES)
    if cm.size == 0: return

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax, xticklabels=class_names, yticklabels=class_names)
    ax.set(xlabel='Predicted Label', ylabel='True Label', title='Normalised Confusion Matrix')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    save_or_show(fig, save_path, display_in_notebook)

def visualise_augmentations(dataset_path, num_examples=7, img_size=512, display_in_notebook=True):
    import torch, matplotlib.pyplot as plt, numpy as np
    from pathlib import Path
    from gleason_src.trainer import GleasonH5Dataset
    from gleason_src.pspnet import KorniaTrainAugmentation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_aug = max(1, int(num_examples) - 1)

    ds = GleasonH5Dataset(h5_path=Path(dataset_path), split="train")
    idx = np.random.randint(len(ds))
    sample = ds[idx]
    img = sample['image']

    if img.ndim == 3:
        img = img.unsqueeze(0)
    assert img.ndim == 4 and img.shape[-1] == 3, "expected NHWC image tensor"
    img_rep = img.repeat(n_aug, 1, 1, 1).to(device)

    aug = KorniaTrainAugmentation(img_size).to(device).eval()
    with torch.no_grad():
        y = aug(img_rep, alpha=1.0).clamp_(0, 1)

    with torch.no_grad():
        x = img.to(torch.float32)
        if x.max().item() > 1.0:
            x = x.mul_(1.0 / 255.0)
        x = torch.nn.functional.interpolate(
            x.permute(0,3,1,2), size=(img_size, img_size),
            mode="bilinear", align_corners=False
        )[0]

    cols = 1 + n_aug
    plt.figure(figsize=(3*cols, 3))
    ax = plt.subplot(1, cols, 1)
    ax.set_title("Original")
    ax.imshow(x.permute(1,2,0).cpu().numpy())
    ax.axis("off")
    for i in range(n_aug):
        ax = plt.subplot(1, cols, i+2)
        ax.set_title(f"Augmentation #{i+1}")
        ax.imshow(y[i].permute(1,2,0).cpu().numpy())
        ax.axis("off")
    plt.suptitle(f"Training Augmentation Examples ({img_size}x{img_size})")
    plt.tight_layout()

    if display_in_notebook:
        plt.show()
    return {"index": int(idx), "shape": tuple(img.shape[1:4])}

def _get_model_and_data(experiment_name: str, output_dir: Path):
    model_path = output_dir / 'models' / experiment_name / 'best_model.pth'
    if not model_path.exists(): return None, None, None, None, None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', {})
    if not config:
        return None, None, None, None, None

    model = build_model(
        backbone=config.get("backbone", "resnet152"),
        dropout=config.get("dropout_rate", 0.1),
        img_size=config.get("img_size", 512),
        num_classes=config.get("num_classes", 5),
        mode="compiled"
    ).to(device)

    uncompiled_net = model.net._orig_mod if hasattr(model.net, '_orig_mod') else model.net

    uncompiled_net.load_state_dict(checkpoint['model_state_dict'])

    dataset_path = output_dir / 'datasets' / 'gleason_dataset.h5'

    return model, uncompiled_net, dataset_path, device, config

def _sort_and_load(dset, unsorted_indices):
    original_indices = np.array(unsorted_indices, dtype=int).flatten()

    if len(original_indices) == 0:
        return np.empty((0,) + dset.shape[1:], dtype=dset.dtype)

    h5py_indices_to_load = np.unique(original_indices)

    loaded_data_map = {idx: data for idx, data in zip(h5py_indices_to_load, dset[h5py_indices_to_load])}

    output_array = np.array([loaded_data_map[idx] for idx in original_indices])

    return output_array

def visualise_prediction_extremes(experiment_name: str, output_dir: Path, num_samples: int = 3, metric: Optional[str] = None, save_path: Optional[Path] = None, display_in_notebook: bool = False):
    eval_path = output_dir / 'models' / experiment_name / 'evaluation' / 'detailed_results.pkl'
    if not eval_path.exists(): print(f"Evaluation results for '{experiment_name}' not found."); return
    with open(eval_path, 'rb') as f: detailed_results = pickle.load(f)

    model, _, dataset_path, device, config = _get_model_and_data(experiment_name, output_dir)
    if model is None: print(f"Model for '{experiment_name}' not found."); return

    if metric is None:
        metric = config.get('objective_metric', 'grade_aware_williams_index')
        print(f"No metric specified, using objective metric from config: '{metric}'")

    scores = detailed_results['raw'].get(metric)
    if scores is None or scores.size == 0: print(f"Metric '{metric}' not found in results."); return

    with h5py.File(dataset_path, 'r') as f:
        individual_masks = f['test']['individual_masks'][:]
        consensus_masks = torch.mode(torch.from_numpy(individual_masks), dim=1).values.numpy()
        non_bg_mask = np.any(consensus_masks > 0, axis=(1, 2))

    original_indices = np.arange(len(scores))
    valid_indices_original = original_indices[non_bg_mask]
    valid_scores = scores[non_bg_mask]

    if len(valid_scores) < num_samples * 2:
        return

    img_size = config.get('img_size', 512)
    val_augmenter = KorniaValidationAugmentation(img_size).to(device)

    sorted_filtered_indices = np.argsort(valid_scores)
    indices_to_show_filtered = {
        "Best": sorted_filtered_indices[-num_samples:][::-1],
        "Worst": sorted_filtered_indices[:num_samples]
    }
    indices_to_show = {
        "Best": valid_indices_original[indices_to_show_filtered["Best"]],
        "Worst": valid_indices_original[indices_to_show_filtered["Worst"]]
    }

    with h5py.File(dataset_path, 'r') as f:
        test_group = f['test']
        for case, unsorted_indices in indices_to_show.items():
            if len(unsorted_indices) == 0: continue

            images_np = _sort_and_load(test_group['images'], unsorted_indices)
            masks_np = _sort_and_load(test_group['individual_masks'], unsorted_indices)

            batch_raw = torch.from_numpy(images_np).to(device)

            with torch.no_grad():
                batch_imgs = val_augmenter(batch_raw)
                preds, _ = model(batch_imgs)
                preds = preds.argmax(dim=1).cpu().numpy()

            fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples), squeeze=False)
            fig.suptitle(f"{case} Predictions by {metric.replace('_', ' ').title()}", fontsize=16)

            for i, (sample_idx, pred_mask) in enumerate(zip(unsorted_indices, preds)):
                gt_mask = torch.mode(torch.from_numpy(masks_np[i]), dim=0).values.numpy()
                axes[i, 0].imshow(images_np[i]); axes[i, 0].set_title(f'Input (Sample {sample_idx})'); axes[i, 0].axis('off')
                axes[i, 1].imshow(gt_mask, cmap='viridis', vmin=0, vmax=4); axes[i, 1].set_title('Ground Truth (Consensus)'); axes[i, 1].axis('off')
                axes[i, 2].imshow(pred_mask, cmap='viridis', vmin=0, vmax=4); axes[i, 2].set_title(f'Prediction ({metric}: {scores[sample_idx]:.3f})'); axes[i, 2].axis('off')

            patches = [mpatches.Patch(color=plt.get_cmap('viridis')(i/4), label=name) for i, name in enumerate(CLASS_NAMES)]
            fig.legend(handles=patches, bbox_to_anchor=(0.5, 0.01), loc='lower center', ncol=len(CLASS_NAMES))

            final_save_path = None
            if save_path:
                final_save_path = save_path.with_name(f"{save_path.stem}_{case.lower()}{save_path.suffix}")
            plt.tight_layout(rect=[0, 0.05, 1, 0.95]); save_or_show(fig, final_save_path, display_in_notebook=display_in_notebook)

def visualise_uncertainty_extremes(experiment_name: str, output_dir: Path, num_samples: int = 3, save_path: Optional[Path] = None, display_in_notebook: bool = False):
    from tqdm import tqdm
    
    model, uncompiled_net, dataset_path, device, config = _get_model_and_data(experiment_name, output_dir)
    if model is None: 
        print(f"Model for '{experiment_name}' not found.")
        return

    img_size = config.get('img_size', 512)
    val_augmenter = KorniaValidationAugmentation(img_size).to(device)

    with h5py.File(dataset_path, 'r') as f:
        individual_masks = f['test']['individual_masks'][:]
        consensus_masks = torch.mode(torch.from_numpy(individual_masks), dim=1).values.numpy()
        non_bg_mask = np.any(consensus_masks > 0, axis=(1, 2))

    valid_indices = np.where(non_bg_mask)[0]
    
    if len(valid_indices) < num_samples:
        print(f"Not enough tissue samples found: {len(valid_indices)}")
        return

    print(f"Computing uncertainty for {len(valid_indices)} tissue samples...")
    
    uncertainty_scores = []
    batch_size = 8  
    
    with h5py.File(dataset_path, 'r') as f:
        test_group = f['test']
        
        for i in tqdm(range(0, len(valid_indices), batch_size), desc="Computing uncertainty"):
            batch_indices = valid_indices[i:i + batch_size]
            images_np = _sort_and_load(test_group['images'], batch_indices)
            batch_raw = torch.from_numpy(images_np).to(device)
            
            with torch.no_grad():
                batch_imgs = val_augmenter(batch_raw)
                batch_imgs_norm = model.norm(batch_imgs)
                mc_n = int(config.get('mc_samples', 10))
                _, uncertainty_map = uncompiled_net.monte_carlo_predict(batch_imgs_norm, n_samples=mc_n)
                
                for j, idx in enumerate(batch_indices):
                    consensus_mask = consensus_masks[idx]
                    tissue_mask = (consensus_mask > 0).astype(np.float32)
                    uncertainty_np = uncertainty_map[j].cpu().numpy()
                    
                    if tissue_mask.sum() > 0:
                        avg_uncertainty = (uncertainty_np * tissue_mask).sum() / tissue_mask.sum()
                    else:
                        avg_uncertainty = 0.0
                    
                    uncertainty_scores.append(avg_uncertainty)
    
    uncertainty_scores = np.array(uncertainty_scores)
    
    print(f"Uncertainty statistics:")
    print(f"  Min: {uncertainty_scores.min():.6f}")
    print(f"  Max: {uncertainty_scores.max():.6f}")
    print(f"  Mean: {uncertainty_scores.mean():.6f}")
    print(f"  Median: {np.median(uncertainty_scores):.6f}")
    print(f"  75th percentile: {np.percentile(uncertainty_scores, 75):.6f}")
    print(f"  90th percentile: {np.percentile(uncertainty_scores, 90):.6f}")
    
    high_threshold = np.percentile(uncertainty_scores, 80)
    print(f"  Using high uncertainty threshold: > {high_threshold:.6f}")
    
    uncertain_mask = uncertainty_scores > high_threshold
    uncertain_filtered_idx = np.where(uncertain_mask)[0]
    
    print(f"  Found {len(uncertain_filtered_idx)} high uncertainty samples")
    
    if len(uncertain_filtered_idx) == 0:
        print("No high uncertainty samples found")
        return
    
    uncertain_to_show = uncertain_filtered_idx[:num_samples] if len(uncertain_filtered_idx) >= num_samples else uncertain_filtered_idx
    sample_indices = valid_indices[uncertain_to_show]

    with h5py.File(dataset_path, 'r') as f:
        test_group = f['test']
        
        images_np = _sort_and_load(test_group['images'], sample_indices)
        masks_np = _sort_and_load(test_group['individual_masks'], sample_indices)
        batch_raw = torch.from_numpy(images_np).to(device)

        with torch.no_grad():
            batch_imgs = val_augmenter(batch_raw)
            batch_imgs_norm = model.norm(batch_imgs)
            mc_n = int(config.get('mc_samples', 10))
            mean_pred, uncertainty_map = uncompiled_net.monte_carlo_predict(batch_imgs_norm, n_samples=mc_n)
            pred_masks = mean_pred.argmax(dim=1).cpu().numpy()
            uncertainty_maps_np = uncertainty_map.cpu().numpy()

        actual_num_samples = len(sample_indices)
        fig, axes = plt.subplots(actual_num_samples, 4, figsize=(16, 4 * actual_num_samples), squeeze=False)
        fig.suptitle(f"High Uncertainty Predictions", fontsize=16)

        for i, (sample_idx, pred_mask, uncertainty) in enumerate(zip(sample_indices, pred_masks, uncertainty_maps_np)):
            gt_mask = torch.mode(torch.from_numpy(masks_np[i]), dim=0).values.numpy()
            
            axes[i, 0].imshow(images_np[i])
            axes[i, 0].set_title(f'Input (Sample {sample_idx})')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(gt_mask, cmap='viridis', vmin=0, vmax=4)
            axes[i, 1].set_title('Ground Truth (Consensus)')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_mask, cmap='viridis', vmin=0, vmax=4)
            axes[i, 2].set_title('Model Prediction')
            axes[i, 2].axis('off')

            tissue = (gt_mask > 0).astype(np.float32)
            avg_u_tissue = (uncertainty * tissue).sum() / max(tissue.sum(), 1)
            axes[i, 3].set_title(f'Uncertainty (Avg: {avg_u_tissue:.4f})')
            axes[i, 3].axis('off')
            im = axes[i, 3].imshow(uncertainty, cmap='inferno')
            fig.colorbar(im, ax=axes[i, 3])

        patches = [mpatches.Patch(color=plt.get_cmap('viridis')(k/4), label=name) for k, name in enumerate(CLASS_NAMES)]
        fig.legend(handles=patches, bbox_to_anchor=(0.5, 0.01), loc='lower center', ncol=len(CLASS_NAMES))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        save_or_show(fig, save_path, display_in_notebook=display_in_notebook)
        
def visualise_disagreement_and_predictions(experiment_name: str, output_dir: Path, num_samples: int = 3, save_dir: Optional[Path] = None, display_in_notebook: bool = False):
    model, _, dataset_path, device, config = _get_model_and_data(experiment_name, output_dir)
    if model is None: print(f"Model for '{experiment_name}' not found."); return

    img_size = config.get('img_size', 512)
    val_augmenter = KorniaValidationAugmentation(img_size).to(device)

    with h5py.File(dataset_path, 'r') as f:
        test_group = f['test']
        if 'disagreement_map_std' not in test_group: print("Disagreement maps not found."); return

        individual_masks_all = test_group['individual_masks'][:]
        consensus_masks = torch.mode(torch.from_numpy(individual_masks_all), dim=1).values.numpy()
        non_bg_mask = np.any(consensus_masks > 0, axis=(1, 2))

        disagreement_scores = np.mean(test_group['disagreement_map_std'][:], axis=(1, 2, 3))

        original_indices = np.arange(len(disagreement_scores))
        valid_indices_original = original_indices[non_bg_mask]
        valid_scores = disagreement_scores[non_bg_mask]

        if len(valid_scores) < num_samples:
            return

        p_names = json.loads(test_group.attrs.get('pathologists', '[]'))

        sorted_filtered_indices = np.argsort(valid_scores)[-num_samples:][::-1]
        unsorted_indices = valid_indices_original[sorted_filtered_indices]

        images_np = _sort_and_load(test_group['images'], unsorted_indices)
        batch_raw = torch.from_numpy(images_np).to(device)

        with torch.no_grad():
            batch_imgs = val_augmenter(batch_raw)
            preds, _ = model(batch_imgs)
            preds = preds.argmax(dim=1).cpu().numpy()

        all_individual_masks = _sort_and_load(test_group['individual_masks'], unsorted_indices)
        disagreement_maps = _sort_and_load(test_group['disagreement_map_std'], unsorted_indices)


        for i, (sample_idx, pred_mask) in enumerate(zip(unsorted_indices, preds)):
            img_np = images_np[i]
            masks = all_individual_masks[i]
            individual_masks = {p: mask for p, mask in zip(p_names, masks) if mask.any()}
            if not individual_masks: continue

            disagreement_map = disagreement_maps[i].squeeze()
            heatmap = cv2.applyColorMap((disagreement_map / (disagreement_map.max() + 1e-8) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            highlighted_img = cv2.addWeighted(img_np, 0.6, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), 0.4, 0)

            fig, axes = plt.subplots(1, 3 + len(individual_masks), figsize=((3 + len(individual_masks)) * 4, 4.5))
            fig.suptitle(f'High-Disagreement Sample ({sample_idx}) | Score: {disagreement_scores[sample_idx]:.3f}', fontsize=16)

            axes[0].imshow(img_np); axes[0].set_title("Original Image"); axes[0].axis('off')
            axes[1].imshow(highlighted_img); axes[1].set_title("Disagreement Heatmap"); axes[1].axis('off')
            axes[2].imshow(pred_mask, cmap='viridis', vmin=0, vmax=4); axes[2].set_title("Model Prediction"); axes[2].axis('off')
            for j, (p_name, p_mask) in enumerate(sorted(individual_masks.items())):
                ax = axes[3 + j]; ax.imshow(p_mask, cmap='viridis', vmin=0, vmax=4)
                ax.set_title(_format_pathologist_name(p_name)); ax.axis('off')

            patches = [mpatches.Patch(color=plt.get_cmap('viridis')(k/4), label=name) for k, name in enumerate(CLASS_NAMES)]
            fig.legend(handles=patches, bbox_to_anchor=(0.5, 0.01), loc='lower center', ncol=len(CLASS_NAMES))
            
            final_save_path = None
            if save_dir:
                final_save_path = save_dir / f"disagreement_sample_{sample_idx}.png"
            plt.tight_layout(rect=[0, 0.05, 1, 0.95]); save_or_show(fig, final_save_path, display_in_notebook=display_in_notebook)