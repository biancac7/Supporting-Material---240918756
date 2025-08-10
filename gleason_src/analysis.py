import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple

from optuna.visualization import plot_optimization_history as optuna_plot_history
from optuna.visualization import plot_slice as optuna_plot_slice

from .visualisations import save_or_show

def _get_study_dataframe(study: optuna.Study) -> pd.DataFrame:
    records = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        record = {'score': trial.value}
        record.update(trial.params)
        records.append(record)
    return pd.DataFrame(records)

def _plot_categorical_overview(df: pd.DataFrame, param: str, ax: plt.Axes):
    order = df.groupby(param)['score'].median().sort_values(ascending=False).index
    sns.boxplot(x=param, y='score', data=df, ax=ax, order=order, palette='viridis')
    ax.set_title(f'Score Distribution by {param}')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

def _plot_numerical_overview(df: pd.DataFrame, param: str, ax: plt.Axes):
    sns.scatterplot(x=param, y='score', data=df, ax=ax, hue='main_loss_name', alpha=0.7, palette='plasma')
    ax.set_title(f'Score vs. {param}')
    if 'decay' in param or 'lr' in param or 'learning_rate' in param:
        ax.set_xscale('log')
    ax.legend(title='Loss Function', loc='lower right')

def _prepare_data_for_modelling(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df_clean = df.dropna(subset=['score']).copy()
    
    X = df_clean.drop('score', axis=1)
    y = df_clean['score']
    
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col].fillna(0, inplace=True) 
        elif pd.api.types.is_object_dtype(X[col]):
            X[col].fillna("N/A", inplace=True)
            
    categorical_features = [col for col in X.columns if X[col].dtype == 'object']
    if categorical_features:
        X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    else:
        X_encoded = X
    
    return X_encoded, y, categorical_features

def plot_optuna_history(study: optuna.Study, save_path: Optional[Path] = None, display_in_notebook: bool = False):
    fig = optuna_plot_history(study)
    if save_path:
        fig.write_image(str(save_path))
    if display_in_notebook:
        fig.show()

def plot_optuna_slice(study: optuna.Study, save_path: Optional[Path] = None, display_in_notebook: bool = False):
    fig = optuna_plot_slice(study)
    if save_path:
        fig.write_image(str(save_path))
    if display_in_notebook:
        fig.show()

def plot_hyperparameter_overview(study: optuna.Study, save_path: Optional[Path] = None, display_in_notebook: bool = False):
    if not study.trials:
        print("Study contains no trials to analyse.")
        return

    df = _get_study_dataframe(study)
    if df.empty:
        print("No completed trials found in the study.")
        return

    params = list(study.best_params.keys())
    
    categorical_params = [p for p in params if df[p].dtype == 'object' or df[p].nunique() < 10]
    numerical_params = [p for p in params if p not in categorical_params]
    
    all_params = categorical_params + numerical_params
    n_params = len(all_params)
    
    n_cols = 2
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 6), squeeze=False)
    axes = axes.flatten()

    for i, param in enumerate(all_params):
        if param in categorical_params:
            _plot_categorical_overview(df, param, axes[i])
        else:
            _plot_numerical_overview(df, param, axes[i])

    for i in range(n_params, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(f'Hyperparameter Overview for Study "{study.study_name}"', fontsize=20, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_or_show(fig, save_path, display_in_notebook=display_in_notebook)

def analyse_pdp(study: optuna.Study, save_dir: Path, display_in_notebook: bool = False):
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        print("LGBM not found, skipping Surrogate Model/PDP analysis. Install with: pip install lightgbm")
        return

    df = _get_study_dataframe(study)
    if df.empty or len(df) < 10: 
        print(f"Insufficient data for PDP analysis: {len(df)} trials (need at least 10)")
        return
        
    X, y, _ = _prepare_data_for_modelling(df)
    if X.empty:
        print("Warning: Could not prepare data for surrogate model analysis (X is empty).")
        return
    
    model = LGBMRegressor(
        random_state=42,
        n_estimators=20,     
        max_depth=3,         
        min_child_samples=1,
        min_data_in_leaf=1, 
        verbose=-1            
    )
    
    try:
        model.fit(X, y)
    except:
        print("Failed to train surrogate model, creating simple feature importance plot instead")
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            correlations = []
            for col in numeric_cols:
                corr = np.corrcoef(X[col], y)[0,1]
                if not np.isnan(corr):
                    correlations.append((col, abs(corr)))
            
            if correlations:
                correlations.sort(key=lambda x: x[1], reverse=True)
                fig, ax = plt.subplots(figsize=(10, 6))
                cols, corrs = zip(*correlations[:8])
                ax.barh(cols, corrs)
                ax.set_title('Parameter Correlations with Score')
                ax.set_xlabel('Absolute Correlation')
                save_or_show(fig, save_dir / "correlations.png", display_in_notebook=display_in_notebook)
        return
    
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = importances.head(min(6, len(X.columns))).index.tolist()
    
    if not top_features:
        print("No important features found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    importances.head(8).plot(kind='barh', ax=ax)
    ax.set_title('Feature Importances (Surrogate Model)')
    ax.set_xlabel('Importance')
    
    save_or_show(fig, save_dir / "feature_importance.png", display_in_notebook=display_in_notebook)

def analyse_gp(study: optuna.Study, save_dir: Path, display_in_notebook: bool = False):
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    except ImportError:
        print("scikit-learn not found, skipping Gaussian Process analysis. Install with: pip install scikit-learn")
        return
        
    df = _get_study_dataframe(study)
    if df.empty: return
    X, y, _ = _prepare_data_for_modelling(df)
    if X.empty:
        print("Warning: Could not prepare data for Gaussian Process analysis (X is empty).")
        return
    
    kernel = Matern(length_scale=np.ones(X.shape[1]), nu=2.5) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, random_state=42)
    gp.fit(X, y)

    length_scales = gp.kernel_.k1.length_scale if hasattr(gp.kernel_, 'k1') else gp.kernel_.length_scale
    importances = pd.Series(1.0 / np.asarray(length_scales), index=X.columns).sort_values(ascending=False)
    top_features = importances.head(4).index.tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle('Gaussian Process: Hyperparameter Effects', fontsize=20)

    for i, feature in enumerate(top_features):
        X_probe = X.median().to_frame().T
        X_probe = pd.concat([X_probe]*100, ignore_index=True)
        X_probe[feature] = np.linspace(X[feature].min(), X[feature].max(), 100)
        
        mean, std = gp.predict(X_probe, return_std=True)
        
        axes[i].plot(X_probe[feature], mean, color='mediumblue', label='Mean Predicted Score')
        axes[i].fill_between(X_probe[feature], mean - 1.96 * std, mean + 1.96 * std, color='cornflowerblue', alpha=0.3, label='95% Confidence Interval')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Predicted Score')
        axes[i].set_title(f'Effect of {feature}')
        axes[i].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_or_show(fig, save_dir / "gp_analysis.png", display_in_notebook=display_in_notebook)