import optuna
import torch
import h5py, traceback, pprint, logging
from pathlib import Path
from typing import Dict, List
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from .pipeline import run_training_pipeline
from .trainer import TRIAL_PRUNED_EXCEPTION

def cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch._dynamo.reset()

def suggest_loss_config(trial: optuna.trial.Trial, prefix: str, task_type: str = 'segmentation') -> Dict:
    loss_config: Dict[str, any] = {'weighting_schemes': []}

    if task_type == 'segmentation':
        loss_name = trial.suggest_categorical(f'{prefix}_loss_name', ['emd', 'emd', 'focal', 'wce'])
    else:
        loss_name = trial.suggest_categorical(f'{prefix}_loss_name', ['wce', 'focal'])
    loss_config['name'] = loss_name

    if loss_name == 'focal':
        loss_config['gamma'] = trial.suggest_float(f'{prefix}_focal_gamma', 2.0, 5.0)
    elif loss_name == 'emd':
        loss_config['cost_type'] = trial.suggest_categorical(f'{prefix}_emd_cost_type', ['L1', 'L2'])
    elif loss_name == 'williams_index':
        loss_config['epsilon'] = trial.suggest_float(f'{prefix}_williams_epsilon', 1e-12, 1e-8, log=True)
        loss_config['temperature'] = trial.suggest_float(f'{prefix}_williams_temp', 0.5, 2.0)
    elif loss_name == 'wce':
        pass

    weighting_choice = trial.suggest_categorical(f'{prefix}_weighting_scheme', ['none', 'disagreement', 'scaled', 'both'])

    if weighting_choice in ['disagreement', 'both']:
        scheme = {
            'type': 'disagreement',
            'metric': trial.suggest_categorical(f'{prefix}_disagreement_metric', ['std', 'mad']),
            'epsilon': trial.suggest_float(f'{prefix}_disagreement_epsilon', 1e-7, 1e-5, log=True)
        }
        loss_config['weighting_schemes'].append(scheme)

    if weighting_choice in ['scaled', 'both']:
        scheme = {
            'type': 'scaled',
            'priority_scale_factor': trial.suggest_float(f'{prefix}_scale_factor', 1.5, 5.0)
        }
        loss_config['weighting_schemes'].append(scheme)

    reducer_type = trial.suggest_categorical(f'{prefix}_reducer', ['mean', 'priority_split'])
    if reducer_type == 'priority_split':
        loss_config['reducer'] = {
            'type': 'priority_split',
            'priority_weight': trial.suggest_float(f'{prefix}_priority_weight', 0.55, 0.95, step=0.05)
        }
    elif reducer_type == 'mean':
        loss_config['reducer'] = {'type': 'mean'}

    return loss_config

def _validate_hpo_config(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise TypeError(f"config must be dict, got {type(cfg)}")

    loss = cfg.get('loss')
    if not isinstance(loss, dict):
        raise TypeError(f"config['loss'] must be dict, got {type(loss)}")

    main = loss.get('main')
    if not isinstance(main, dict):
        raise TypeError(f"config['loss']['main'] must be dict, got {type(main)}: {repr(main)[:200]}")

    if 'name' not in main:
        raise KeyError("config['loss']['main'] missing 'name'")

    ws = main.get('weighting_schemes', [])
    if ws is not None and not isinstance(ws, list):
        raise TypeError(f"config['loss']['main']['weighting_schemes'] must be list, got {type(ws)}")
    for i, s in enumerate(ws or []):
        if not isinstance(s, dict) or 'type' not in s:
            raise TypeError(f"weighting_schemes[{i}] must be dict with 'type', got: {repr(s)[:120]}")

    if loss.get('aux_weight', 0) > 1e-6:
        aux = loss.get('aux')
        if not isinstance(aux, dict):
            raise TypeError(f"config['loss']['aux'] must be dict when aux_weight>0, got {type(aux)}")

def objective(trial: optuna.trial.Trial, experiment_config: Dict, datasets_dir: Path, output_dir: Path):
    main_params = suggest_loss_config(trial, 'main', task_type='segmentation')

    loss_config = {
        'main'      : main_params,
        'aux_weight': trial.suggest_float('aux_weight', 0.0, 0.4, step=0.05),
    }

    if loss_config['aux_weight'] > 1e-6:
        loss_config['aux'] = suggest_loss_config(trial, 'aux', task_type='classification')
    else:
        loss_config['aux'] = {}

    loss_config['class_weights'] = experiment_config.get('base_class_weights_tensor')

    config = {
        'num_epochs'                 : experiment_config.get('num_epochs', 25),
        'num_classes'                : experiment_config.get('num_classes', 5),
        'img_size'                   : experiment_config.get('target_size', (512, 512))[0],
        'batch_size'                 : experiment_config.get('batch_size', 16),
        'backbone'                   : experiment_config.get('backbone', 'resnet152'),
        'loss'                       : loss_config,
        'name'                       : experiment_config.get('name', f"trial_{trial.number}"),
        'learning_rate'              : trial.suggest_float('learning_rate', 5e-6, 5e-4, log=True),
        'dropout_rate'               : trial.suggest_float('dropout_rate', 0.1, 0.4, step=0.05),
        'save_model_weights'         : experiment_config.get('save_model_weights', False),
        'num_workers'                : experiment_config.get('num_workers', 0),
        'gradient_accumulation_steps': experiment_config.get('gradient_accumulation_steps', 1),
        'weight_decay'               : trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
        'scheduler_patience'         : experiment_config.get('scheduler_patience', 3),
        'early_stopping_patience'    : experiment_config.get('early_stopping_patience', 10),
        'objective_metric'           : experiment_config.get('objective_metric', 'williams_index')
    }

    try:
        _validate_hpo_config(config)
        results = run_training_pipeline(datasets_dir, output_dir, config['name'], config,
                                        verbose=False, trial=trial, trial_num=trial.number, is_hpo_run=True)

        trial.set_user_attr('best_epoch', results.get('best_epoch', -1))
        for k, v in results.get('best_metrics', {}).items():
            trial.set_user_attr(f'best_{k}', v)

        objective_metric_name = config['objective_metric']
        score = results.get('best_metrics', {}).get(objective_metric_name, -1.0)
        return float(score)

    except TRIAL_PRUNED_EXCEPTION:
        raise

    except Exception:
        print("\n===== HPO CONFIG DUMP =====")
        print(pprint.pformat(config, width=120))
        print("TYPES:", {k: type(v).__name__ for k, v in config.items()})
        print("LOSS TYPES:", type(config.get('loss')).__name__,
              " main:", type(config.get('loss',{}).get('main')).__name__,
              " aux:",  type(config.get('loss',{}).get('aux')).__name__)
        traceback.print_exc()
        raise

    finally:
        cleanup()

def run_optuna_study(
    n_trials: int,
    exp_config: Dict,
    datasets_dir: Path,
    output_dir: Path,
    study_name: str,
    db_path: str
) -> optuna.Study:
    from tqdm import tqdm

    optuna.logging.get_logger("optuna").setLevel(logging.WARNING)

    h5_path = datasets_dir / 'gleason_dataset.h5'
    if h5_path.exists():
        with h5py.File(h5_path, 'r') as f:
            if 'train' in f and 'class_weights' in f['train']:
                exp_config['base_class_weights_tensor'] = torch.as_tensor(f['train/class_weights'][:], dtype=torch.float32)
    else:
        print(f"WARNING: Training data not found at {h5_path}. Running HPO without pre-computed weights.")

    objective_metric_name = exp_config.get('objective_metric', 'williams_index')
    direction = 'minimize' if 'loss' in objective_metric_name else 'maximize'
    print(f"Starting Optuna study. Objective: {direction} '{objective_metric_name}'.")

    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=4, interval_steps=1)
    sampler = TPESampler(n_startup_trials=20, seed=42, multivariate=True, warn_independent_sampling=False)

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=db_path,
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler
    )

    n_completed = 0
    n_pruned    = 0
    try:
        best_score = study.best_value
    except ValueError:
        best_score = float('inf') if direction == 'minimize' else -float('inf')

    with tqdm(total=n_trials, desc="Optimising", position=0, leave=True) as pbar:

        def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
            nonlocal n_completed, n_pruned, best_score

            pbar.update(1)

            if trial.state == optuna.trial.TrialState.COMPLETE:
                n_completed += 1
                if trial.value is not None:
                    if direction == 'maximize' and trial.value > best_score:
                        best_score = trial.value
                    elif direction == 'minimize' and trial.value < best_score:
                        best_score = trial.value
            elif trial.state == optuna.trial.TrialState.PRUNED:
                n_pruned += 1

            pbar.set_description_str(
                f"Optimising (Best Score: {best_score:.4f} | "
                f"Completed: {n_completed}, Pruned: {n_pruned})"
            )
            pbar.refresh()

        try:
            study.optimize(
                lambda trial: objective(trial, exp_config, datasets_dir, output_dir),
                n_trials=n_trials,
                gc_after_trial=True,
                callbacks=[callback],
                show_progress_bar=False
            )
        except Exception as e:
            pbar.close()
            print(f"\nOptimisation stopped unexpectedly: {e}")
            raise e
        finally:
            optuna.logging.get_logger("optuna").setLevel(logging.INFO)

    return study