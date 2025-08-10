import h5py
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
import cv2
from itertools import combinations
import albumentations as A
from tqdm import tqdm

REMAP_LUT = np.array([0, 1, 0, 2, 3, 4, 1], dtype=np.uint8)
CLASS_NAMES = ['Background', 'Benign', 'Gleason_3', 'Gleason_4', 'Gleason_5']
BATCH_SIZE = 32

CLINICAL_CLASS_INDICES = [1, 2, 3, 4]

class FileRegistry:
    def __init__(self):
        self.df = pd.DataFrame(columns=['path', 'category', 'slide', 'core', 'pathologist', 'file_type'])

    def add_files(self, file_info_list: List[Dict]):
        if not file_info_list: return
        self.df = pd.concat([self.df, pd.DataFrame(file_info_list)], ignore_index=True).astype({
            'slide': 'int16', 'core': 'int16', 'category': 'category',
            'pathologist': 'category', 'file_type': 'category'
        })

    def get_common_cores(self, pathologists: List[str], min_pathologists: Optional[int] = None) -> pd.DataFrame:
        mask_df = self.df[(self.df['file_type'] == 'mask') & (self.df['pathologist'].isin(pathologists))]
        coverage = mask_df.groupby(['slide', 'core'])['pathologist'].nunique()
        valid_cores_idx = coverage[coverage >= (min_pathologists or len(pathologists))].index
        return pd.DataFrame(valid_cores_idx.tolist(), columns=['slide', 'core'])

    def get_annotations_batch(self, cores_df: pd.DataFrame, pathologists: List[str]) -> List[Dict]:
        mask_df = self.df[(self.df['file_type'] == 'mask') & (self.df['pathologist'].isin(pathologists))]
        pivoted = cores_df.merge(mask_df, on=['slide', 'core']).pivot_table(
            index=['slide', 'core'], columns='pathologist', values='path', aggfunc='first'
        ).reset_index()
        img_df = self.df[self.df['file_type'] == 'image'][['slide', 'core', 'path']].rename(columns={'path': 'image_path'})
        return pivoted.merge(img_df, on=['slide', 'core']).to_dict('records')

@lru_cache(maxsize=None)
def parse_filename(filepath: str) -> Optional[Tuple[int, int]]:
    match = re.search(r'slide(\d+)_core(\d+)', Path(filepath).stem)
    return (int(match.group(1)), int(match.group(2))) if match else None

def scan_directory(source_dir: Path, pathologists: List[str]) -> FileRegistry:
    registry, file_info = FileRegistry(), []
    for f in source_dir.glob('Train Imgs/**/*.jpg'):
        if parsed := parse_filename(str(f)):
            file_info.append({'path': str(f), 'slide': parsed[0], 'core': parsed[1], 'pathologist': 'N/A', 'file_type': 'image', 'category': 'image'})
    for p_name in pathologists:
        for f in (source_dir / p_name).glob('**/*_classimg_nonconvex.png'):
            if parsed := parse_filename(str(f)):
                file_info.append({'path': str(f), 'slide': parsed[0], 'core': parsed[1], 'pathologist': p_name, 'file_type': 'mask', 'category': p_name})
    registry.add_files(file_info)
    return registry

class BatchImageLoader:
    def __init__(self, cache_size: int = 256):
        self._cache = lru_cache(maxsize=cache_size)(self._load_image)

    def _load_image(self, path: str, target_size: Optional[Tuple[int, int]], grayscale: bool):
        if not path: return np.zeros((*target_size, 3) if not grayscale else target_size, dtype=np.uint8)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED if grayscale else cv2.IMREAD_COLOR)
        if img is None: return np.zeros((*target_size, 3) if not grayscale else target_size, dtype=np.uint8)

        if not grayscale: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if target_size and img.shape[:2] != target_size:
            interp = cv2.INTER_AREA if np.prod(img.shape[:2]) > np.prod(target_size) else cv2.INTER_LINEAR
            img = cv2.resize(img, target_size, interpolation=interp)
        return img

    def load_batch(self, paths: List[str], target_size: Optional[Tuple[int, int]] = None, grayscale: bool = False) -> np.ndarray:
        return np.stack([self._cache(p, target_size, grayscale) for p in paths])

def compute_agreement_matrices_vectorised(registry: FileRegistry, pathologists: List[str], target_size: Tuple[int, int] = (256, 256)) -> Dict[str, np.ndarray]:
    n_p, n_c = len(pathologists), len(CLASS_NAMES)
    total_sums = np.zeros((n_c, n_p, n_p))
    total_counts = np.zeros((n_c, n_p, n_p))
    common_cores = registry.get_common_cores(pathologists)
    if common_cores.empty: return {}

    loader, batch_data = BatchImageLoader(), registry.get_annotations_batch(common_cores, pathologists)

    for i in tqdm(range(0, len(batch_data), BATCH_SIZE), desc=f"{'Computing agreement':<26}"):
        batch = batch_data[i:i + BATCH_SIZE]
        mask_paths = [[s.get(p) for s in batch] for p in pathologists]

        all_masks = np.stack([loader.load_batch(paths, target_size, True) for paths in mask_paths], axis=1)
        remapped = REMAP_LUT[np.clip(all_masks, 0, len(REMAP_LUT) - 1)]

        m1 = np.expand_dims(remapped, 2)
        m2 = np.expand_dims(remapped, 1)
        pixel_agreement_matrix = (m1 == m2)

        for cid in range(n_c):
            class_mask = (m1 == cid) | (m2 == cid)

            total_sums[cid] += np.sum(pixel_agreement_matrix & class_mask, axis=(0, 3, 4))
            total_counts[cid] += np.sum(class_mask, axis=(0, 3, 4))

    avg_agreement = {}
    for cid in range(n_c):
        with np.errstate(divide='ignore', invalid='ignore'):
            matrix = np.nan_to_num(total_sums[cid] / total_counts[cid])
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 1.0)
        avg_agreement[CLASS_NAMES[cid]] = matrix
    return avg_agreement

def compute_reliability_weights(agreement_matrices: Dict[str, np.ndarray], pathologists: List[str]) -> Dict[str, Dict[str, float]]:
    weights = {p: {} for p in pathologists}
    for class_name, matrix in agreement_matrices.items():
        reliabilities = (matrix.sum(axis=1) - 1) / (len(pathologists) - 1 if len(pathologists) > 1 else 1)
        for p_name, reliability in zip(pathologists, reliabilities):
            weights[p_name][class_name] = reliability if np.isfinite(reliability) else 1.0
    return weights

def create_soft_labels_vectorised(remapped_masks: np.ndarray, weights: Dict, p_names: List) -> np.ndarray:
    n_p, h, w = remapped_masks.shape
    weight_map = np.ones_like(remapped_masks, dtype=np.float32)
    for p_idx, p_name in enumerate(p_names):
        p_weights = weights.get(p_name, {})
        for c_idx, c_name in enumerate(CLASS_NAMES):
            weight_map[p_idx][remapped_masks[p_idx] == c_idx] = p_weights.get(c_name, 1.0)

    soft_labels = np.zeros((len(CLASS_NAMES), h, w), dtype=np.float32)
    for c_idx in range(len(CLASS_NAMES)):
        soft_labels[c_idx] = np.sum((remapped_masks == c_idx) * weight_map, axis=0)

    total_weight = soft_labels.sum(axis=0)
    soft_labels /= (total_weight + 1e-8); soft_labels[0, total_weight < 1e-8] = 1.0
    return soft_labels

def save_dataset_hdf5(samples: List, weights: Dict, out_path: Path, p_names: List, size: Tuple, split_name: str):
    if not samples: return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_s, n_c, n_p = len(samples), len(CLASS_NAMES), len(p_names)
    loader = BatchImageLoader()

    is_train = split_name == 'train'
    img_chunks = (1, *size, 3)
    lbl_chunks = (1, n_c, *size)
    ind_chunks = (1, n_p, *size)
    dis_chunks = (1, 1, *size)

    compression = None if is_train else 'gzip'

    with h5py.File(out_path, 'a') as f:
        f.rdcc_nbytes = 1024**2 * 16
        f.rdcc_nslots = 1e4

        split_group = f.require_group(split_name)

        dsets = {name: split_group.create_dataset(name, shape, dtype=dtype, chunks=chunks, compression=compression)
                 for name, shape, dtype, chunks in [
                     ('images', (n_s, *size, 3), 'uint8', img_chunks),
                     ('soft_labels', (n_s, n_c, *size), 'float32', lbl_chunks),
                     ('individual_masks', (n_s, n_p, *size), 'uint8', ind_chunks),
                     ('disagreement_map_std', (n_s, 1, *size), 'float32', dis_chunks),
                     ('disagreement_map_mad', (n_s, 1, *size), 'float32', dis_chunks),
                     ('inter_expert_error_map', (n_s, 1, *size), 'float32', dis_chunks),
                     ('inter_expert_emd_map', (n_s, 1, *size), 'float32', dis_chunks),
                 ]}

        for i in tqdm(range(0, n_s, BATCH_SIZE), desc=f"{f'Saving {split_name} split':<26}"):
            batch = samples[i:i + BATCH_SIZE]; slc = slice(i, i + len(batch))
            dsets['images'][slc] = loader.load_batch([s['image_path'] for s in batch], size)

            s_lbls, i_msks, dis_stds, dis_mads, exp_errs, exp_emds = [], [], [], [], [], []
            for sample in batch:
                p_present, masks_loaded = [], []
                all_masks = np.zeros((n_p, *size), dtype=np.uint8)
                for p_idx, p_name in enumerate(p_names):
                    if p_name in sample and pd.notna(sample[p_name]):
                        mask = loader.load_batch([sample[p_name]], size, True)
                        if mask.any():
                            remapped_mask = REMAP_LUT[np.clip(mask[0], 0, len(REMAP_LUT) - 1)]
                            p_present.append(p_name)
                            masks_loaded.append(remapped_mask)
                            all_masks[p_idx] = remapped_mask

                if len(masks_loaded) >= 2:
                    s_lbls.append(create_soft_labels_vectorised(np.stack(masks_loaded), weights, p_present))
                    mask_stack = np.stack(masks_loaded)

                    std_map = np.std(mask_stack, axis=0)
                    max_std = np.sqrt(((n_c - 1)**2) / 4)
                    dis_stds.append((std_map / (max_std + 1e-8))[np.newaxis, :, :])

                    median = np.median(mask_stack, axis=0)
                    mad_map = np.mean(np.abs(mask_stack - median), axis=0)
                    max_mad = (n_c - 1) / 2
                    dis_mads.append((mad_map / (max_mad + 1e-8))[np.newaxis, :, :])

                    pairwise_errors, pairwise_emds = [], []
                    for m1, m2 in combinations(masks_loaded, 2):
                        pairwise_errors.append((m1 != m2).astype(np.float32))

                        m1_oh = np.eye(n_c)[m1.ravel()].reshape((*m1.shape, n_c))
                        m2_oh = np.eye(n_c)[m2.ravel()].reshape((*m2.shape, n_c))
                        m1_cdf = np.cumsum(m1_oh, axis=-1)
                        m2_cdf = np.cumsum(m2_oh, axis=-1)
                        pairwise_emds.append(np.sum(np.abs(m1_cdf - m2_cdf), axis=-1))

                    exp_errs.append(np.mean(pairwise_errors, axis=0)[np.newaxis, :, :])
                    exp_emds.append(np.mean(pairwise_emds, axis=0)[np.newaxis, :, :])

                else:
                    s_lbls.append(np.eye(n_c)[0][:, None, None] if not masks_loaded else create_soft_labels_vectorised(np.stack(masks_loaded), weights, p_present))
                    dis_stds.append(np.zeros((1, *size), dtype=np.float32))
                    dis_mads.append(np.zeros((1, *size), dtype=np.float32))
                    exp_errs.append(np.zeros((1, *size), dtype=np.float32))
                    exp_emds.append(np.zeros((1, *size), dtype=np.float32))

                i_msks.append(all_masks)

            dsets['soft_labels'][slc] = np.stack(s_lbls)
            dsets['individual_masks'][slc] = np.stack(i_msks)
            dsets['disagreement_map_std'][slc] = np.stack(dis_stds)
            dsets['disagreement_map_mad'][slc] = np.stack(dis_mads)
            dsets['inter_expert_error_map'][slc] = np.stack(exp_errs)
            dsets['inter_expert_emd_map'][slc] = np.stack(exp_emds)

        split_group.attrs['pathologists'] = json.dumps(p_names)

def compute_class_weights(hdf5_path: Path, split_name: str) -> np.ndarray:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    with h5py.File(hdf5_path, 'r') as f:
        if split_name not in f: return counts
        labels_dset = f[split_name]['soft_labels']
        for i in range(0, len(labels_dset), BATCH_SIZE):
            unique, c = np.unique(np.argmax(labels_dset[i:i+BATCH_SIZE], axis=1), return_counts=True)
            counts[unique] += c
    return counts.sum() / (len(CLASS_NAMES) * counts + 1e-8)

def compute_sample_weights(hdf5_path: Path, class_weights: np.ndarray, split_name: str) -> np.ndarray:
    if not hdf5_path.exists(): return np.array([])
    num_classes = len(class_weights)

    with h5py.File(hdf5_path, 'r') as f:
        if split_name not in f: return np.array([])
        labels_dset = f[split_name]['soft_labels']
        n_samples, _, h, w = labels_dset.shape
        sample_weights = np.zeros(n_samples, dtype=np.float32)

        pbar = tqdm(range(0, n_samples, BATCH_SIZE), desc=f"{'Assigning sample weights':<26}", leave=False)
        for start_idx in pbar:
            end_idx = min(start_idx + BATCH_SIZE, n_samples)
            batch_slice = slice(start_idx, end_idx)

            soft_labels_batch = labels_dset[batch_slice]
            hard_labels_batch = np.argmax(soft_labels_batch, axis=1)

            batch_counts = (
                hard_labels_batch.reshape(end_idx - start_idx, -1, 1) == np.arange(num_classes)
            ).sum(axis=1)

            proportions = batch_counts / (h * w)
            batch_sample_weights = np.sum(proportions * class_weights, axis=1)
            sample_weights[batch_slice] = batch_sample_weights

    return sample_weights

def split_data_stratified(registry: FileRegistry, p_names: List, ratios=(0.7, 0.15), seed=42, strict=True):
    min_p_count = len(p_names) if strict else 2
    valid_cores = registry.get_common_cores(p_names, min_pathologists=min_p_count)
    samples_df = pd.DataFrame(registry.get_annotations_batch(valid_cores, p_names))
    if samples_df.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    unique_slides = samples_df['slide'].unique()
    np.random.seed(seed); np.random.shuffle(unique_slides)
    train_slides, val_slides, test_slides = np.split(unique_slides, [int(len(unique_slides) * ratios[0]), int(len(unique_slides) * sum(ratios))])

    train_df = samples_df[samples_df['slide'].isin(train_slides)]
    val_df   = samples_df[samples_df['slide'].isin(val_slides)]
    test_df  = samples_df[samples_df['slide'].isin(test_slides)]
    print(f"Data Splitting: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples.")
    return train_df, val_df, test_df