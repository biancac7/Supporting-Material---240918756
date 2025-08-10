from pathlib import Path
from typing import Optional, Dict, Any
import torch
import h5py
import optuna

from gleason_src.optuna_objective import suggest_loss_config

def reconstruct_best_config(study: optuna.Study, base_config: Dict, datasets_dir: Path) -> Optional[Dict[str, Any]]:
    if not study.best_trial:
        print("No best trial found in the study. Cannot reconstruct configuration.")
        return None

    best_params = study.best_trial.params
    dummy_trial = optuna.trial.FixedTrial(best_params)
    
    main_loss_params = suggest_loss_config(dummy_trial, 'main')
    
    aux_loss_params = {}
    if best_params.get('aux_weight', 0.0) > 1e-6:
        aux_loss_params = suggest_loss_config(dummy_trial, 'aux')

    h5_path = datasets_dir / 'gleason_dataset.h5'
    if h5_path.exists():
        with h5py.File(h5_path, 'r') as f:
            if 'train' in f:
                train_group = f['train']
                if 'class_weights' in train_group:
                    base_config['base_class_weights_tensor'] = torch.as_tensor(train_group['class_weights'][:], dtype=torch.float32)
                if 'sample_weights' in train_group:
                    base_config['sample_weights_np'] = train_group['sample_weights'][:]
            
    class_weights_tensor = base_config.get('base_class_weights_tensor')
    
    final_loss_config = {
        'main': main_loss_params,
        'aux': aux_loss_params,
        'aux_weight': best_params.get('aux_weight'),
        'class_weights': class_weights_tensor, 
    }
    
    best_config = {
        'loss'                       : final_loss_config,
        'dropout_rate'               : best_params.get('dropout_rate'),
        'weight_decay'               : best_params.get('weight_decay'),
        'learning_rate'              : best_params.get('learning_rate'),
        
        'backbone'                   : base_config.get('backbone', 'resnet152'),
        'batch_size'                 : base_config.get('batch_size', 16),
        'gradient_accumulation_steps': base_config.get('gradient_accumulation_steps', 1),
        'scheduler_patience'         : base_config.get('scheduler_patience', 3),
        'early_stopping_patience'    : base_config.get('early_stopping_patience', 10),
        'num_workers'                : base_config.get('num_workers', 0),
        'img_size'                   : base_config.get('target_size', (512, 512))[0],

        'num_epochs'                 : base_config.get('num_epochs', 25) * 2, 
        'name'                       : 'best-model-from-search',
        'save_model_weights'         : True,
    }
         
    return best_config