import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

def download_data(folder_id: str, output_dir: Union[str, Path]):
    try:
        import gdown
        output_path = Path(output_dir)
        if not output_path.exists() or not any(output_path.iterdir()):
            print(f"Data not found. Downloading to {output_path}...")
            gdown.download_folder(id=folder_id, output=str(output_path), quiet=False)
            import zipfile
            for zip_path in tqdm(list(output_path.glob('*.zip')), desc="Extracting ZIPs"):
                with zipfile.ZipFile(zip_path, 'r') as zf: zf.extractall(output_path)
                zip_path.unlink()
        else:
            print(f"Data already exists in {output_path}.")
    except ImportError:
        print("Please install gdown to download data: !pip install gdown")
    except Exception as e:
        print(f"An error occurred during data download: {e}")

def set_random_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_or_show(fig: plt.Figure, save_path: Optional[Path] = None, display_in_notebook: bool = False):
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({'figure.facecolor': 'white', 'savefig.dpi': 150, 'savefig.bbox': 'tight'})
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
    if display_in_notebook:
        plt.show()
    plt.close(fig)

def _format_pathologist_name(name: str) -> str:
    return name.replace('_T', '').replace('Maps', 'Pathologist ')