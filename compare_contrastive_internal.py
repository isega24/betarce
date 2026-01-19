"""
Script para comparar métricas dentro del experimento CONTRASTIVE
Compara: Base (sin robustez) vs Posthoc (robx)

También calcula accuracy basado en model1_pred_crisp vs y_test_sample
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# Configurar estilo de seaborn
sns.set_style("whitegrid")
sns.set_palette("husl")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

class Config:
    """Configuración centralizada del script"""
    
    # Rutas base
    RESULTS_BASE = "results"
    EXPERIMENT_DIR = "contrastive_trial/results"
    
    # Explainers base (generan counterfactuals base)
    BASE_EXPLAINERS = ["gs", "dice", "face"]
    
    # Explainers posthoc (aplican robustez a base)
    POSTHOC_EXPLAINERS = ["betarob", "robx"]
    
    # Métricas a comparar
    BASE_METRICS = [
        "base_counterfactual_validity",
        "base_counterfactual_proximityL1",
        "base_counterfactual_proximityL2",
        "base_counterfactual_plausibility",
        "base_counterfactual_discriminative_power"
    ]
    
    ROBUST_METRICS = [
        "robust_counterfactual_validity",
        "robust_counterfactual_proximityL1",
        "robust_counterfactual_proximityL2",
        "robust_counterfactual_plausibility",
        "robust_counterfactual_discriminative_power"
    ]
    
    # Mapeo de nombres legibles para métricas
    METRIC_NAMES = {
        "base_counterfactual_validity": "validity",
        "base_counterfactual_proximityL1": "proximityL1",
        "base_counterfactual_proximityL2": "proximityL2",
        "base_counterfactual_plausibility": "plausibility",
        "base_counterfactual_discriminative_power": "dpow",
        "robust_counterfactual_validity": "validity (robust)",
        "robust_counterfactual_proximityL1": "proximityL1 (robust)",
        "robust_counterfactual_proximityL2": "proximityL2 (robust)",
        "robust_counterfactual_plausibility": "plausibility (robust)",
        "robust_counterfactual_discriminative_power": "dpow (robust)",
        "accuracy": "accuracy"
    }
    
    # Datasets
    DATASETS = [
        "fico",
        "wine_quality",
        "breast_cancer",
        "car_eval",
        "rice",
        "diabetes"
    ]
    
    # Directorio de salida
    OUTPUT_DIR = "results_comparison_contrastive"
    
    # Figsize para gráficos
    FIGSIZE_SINGLE = (12, 6)
    FIGSIZE_MULTI = (16, 10)


# ============================================================================
# FUNCIONES DE CARGA Y PROCESAMIENTO
# ============================================================================

def find_result_files(config: Config) -> List[str]:
    """Encuentra todos los archivos de resultados."""
    base_path = os.path.join(config.RESULTS_BASE, config.EXPERIMENT_DIR)
    result_files = glob.glob(
        os.path.join(base_path, "**", "results", "*.csv"),
        recursive=True
    )
    print(f"[CONTRASTIVE] Encontrados {len(result_files)} archivos de resultados")
    return result_files


def load_contrastive_results(result_files: List[str], config: Config) -> pd.DataFrame:
    """Carga todos los datos de contrastive."""
    dfs = []
    
    for file_path in result_files:
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"Advertencia: No se pudo cargar {file_path}: {e}")
            continue
    
    if not dfs:
        print("Error: No hay datos para contrastive")
        return pd.DataFrame()
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"[CONTRASTIVE] Datos cargados: {len(combined_df)} filas")
    
    return combined_df


def calculate_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula accuracy comparando model1_pred_crisp con y_test_sample.
    
    Args:
        df: DataFrame con predicciones
        
    Returns:
        DataFrame con columna de accuracy agregada
    """
    df_copy = df.copy()
    
    # Calcular accuracy: porcentaje de predicciones correctas
    df_copy['accuracy'] = (df_copy['model1_pred_crisp'] == df_copy['y_test_sample']).astype(int)
    
    return df_copy


def separate_by_robustness(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa datos en:
    - Base: sin posthoc_explainer (NaN)
    - Robust: con posthoc_explainer (robx, betarob)
    
    Args:
        df: DataFrame completo
        
    Returns:
        Tupla (base_df, robust_df)
    """
    
    # Base: sin posthoc_explainer
    base_df = df[df['posthoc_explainer'].isna()].copy()
    base_df['robustness_type'] = 'Base'
    
    # Robust: con posthoc_explainer
    robust_df = df[df['posthoc_explainer'].notna()].copy()
    robust_df['robustness_type'] = robust_df['posthoc_explainer']
    
    print(f"\nDatos separados:")
    print(f"  - Base (sin robustez): {len(base_df)} registros")
    print(f"  - Robust (con posthoc): {len(robust_df)} registros")
    
    return base_df, robust_df


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN (Violin Plots con Seaborn)
# ============================================================================

def plot_metric_violin_by_explainer(
    base_results: pd.DataFrame,
    robust_results: pd.DataFrame,
    metric: str,
    config: Config,
    ax: Optional[plt.Axes] = None
) -> Optional[plt.Axes]:
    """
    Crea violin plot comparativo de una métrica: Base vs Robust.
    """
    
    # Preparar datos
    base_results_copy = base_results.copy()
    base_results_copy['Tipo'] = 'Base'
    
    robust_results_copy = robust_results.copy()
    robust_results_copy['Tipo'] = robust_results_copy['robustness_type']
    
    # Combinar datos
    combined = pd.concat([base_results_copy, robust_results_copy], ignore_index=True)
    
    # Crear gráfico si no existe
    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
    
    # Nombre limpio de la métrica
    metric_name = config.METRIC_NAMES.get(metric, metric.replace('base_counterfactual_', ''))
    
    # Violin plot
    sns.violinplot(
        data=combined,
        x='base_cf_method',
        y=metric,
        hue='Tipo',
        ax=ax,
        palette="Set2",
        inner="box",
        cut=0
    )
    
    # Configurar ejes
    ax.set_xlabel('Explainer Base', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} - Base vs Robustez Posthoc', fontsize=13, fontweight='bold')
    ax.legend(title='Tipo', title_fontsize=11, fontsize=10, loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return ax


def plot_metric_violin_by_dataset(
    base_results: pd.DataFrame,
    robust_results: pd.DataFrame,
    metric: str,
    config: Config,
    ax: Optional[plt.Axes] = None
) -> Optional[plt.Axes]:
    """
    Crea violin plot comparativo de una métrica por dataset.
    """
    
    # Preparar datos
    base_results_copy = base_results.copy()
    base_results_copy['Tipo'] = 'Base'
    
    robust_results_copy = robust_results.copy()
    robust_results_copy['Tipo'] = robust_results_copy['robustness_type']
    
    # Combinar datos
    combined = pd.concat([base_results_copy, robust_results_copy], ignore_index=True)
    
    # Crear gráfico si no existe
    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
    
    # Nombre limpio de la métrica
    metric_name = config.METRIC_NAMES.get(metric, metric.replace('base_counterfactual_', ''))
    
    # Violin plot
    sns.violinplot(
        data=combined,
        x='dataset_name',
        y=metric,
        hue='Tipo',
        ax=ax,
        palette="Set2",
        inner="box",
        cut=0
    )
    
    # Configurar ejes
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} - Base vs Robustez por Dataset', fontsize=13, fontweight='bold')
    ax.legend(title='Tipo', title_fontsize=11, fontsize=10, loc='best', bbox_to_anchor=(1.05, 1))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return ax


def plot_metric_base_vs_robust_by_dataset(
    base_results: pd.DataFrame,
    robust_results: pd.DataFrame,
    base_metric: str,
    robust_metric: str,
    config: Config,
    ax: Optional[plt.Axes] = None
) -> Optional[plt.Axes]:
    """
    Crea violin plot comparativo de una métrica base vs su versión robust,
    para Base y Robust (robx) agrupado por dataset.
    
    Muestra 4 violines por dataset:
    - Base (métrica base)
    - Base (métrica robust)
    - Robx (métrica base)
    - Robx (métrica robust)
    """
    
    # Verificar que las métricas existen
    if base_metric not in base_results.columns or robust_metric not in base_results.columns:
        print(f"Advertencia: Métricas {base_metric} o {robust_metric} no encontradas en base_results")
        return None
    if base_metric not in robust_results.columns or robust_metric not in robust_results.columns:
        print(f"Advertencia: Métricas {base_metric} o {robust_metric} no encontradas en robust_results")
        return None
    
    # Preparar datos - transformar a formato largo para comparar base vs robust
    rows = []
    
    # Base - métrica base
    for _, row in base_results.iterrows():
        rows.append({
            'dataset_name': row['dataset_name'],
            'value': row[base_metric],
            'Variante': 'Base (base metric)'
        })
    
    # Base - métrica robust
    for _, row in base_results.iterrows():
        rows.append({
            'dataset_name': row['dataset_name'],
            'value': row[robust_metric],
            'Variante': 'Base (robust metric)'
        })
    
    # Robust (robx) - métrica base
    for _, row in robust_results.iterrows():
        rows.append({
            'dataset_name': row['dataset_name'],
            'value': row[base_metric],
            'Variante': f"{row['robustness_type']} (base metric)"
        })
    
    # Robust (robx) - métrica robust
    for _, row in robust_results.iterrows():
        rows.append({
            'dataset_name': row['dataset_name'],
            'value': row[robust_metric],
            'Variante': f"{row['robustness_type']} (robust metric)"
        })
    
    combined = pd.DataFrame(rows)
    
    # Crear gráfico si no existe
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))
    
    # Nombre limpio de la métrica (usar el nombre base sin prefijo)
    metric_name = config.METRIC_NAMES.get(base_metric, base_metric.replace('base_counterfactual_', ''))
    
    # Paleta de colores: azul para Base, naranja para Robx, más claro para base metric, más oscuro para robust metric
    palette = {
        'Base (base metric)': '#7fbfff',       # azul claro
        'Base (robust metric)': '#1f77b4',     # azul oscuro
        'robx (base metric)': '#ffbf7f',       # naranja claro
        'robx (robust metric)': '#ff7f0e'      # naranja oscuro
    }
    
    # Orden para las variantes
    hue_order = ['Base (base metric)', 'Base (robust metric)', 'robx (base metric)', 'robx (robust metric)']
    
    # Violin plot
    sns.violinplot(
        data=combined,
        x='dataset_name',
        y='value',
        hue='Variante',
        hue_order=hue_order,
        ax=ax,
        palette=palette,
        inner="box",
        cut=0
    )
    
    # Configurar ejes
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} - Base vs Robust Metric por Tipo y Dataset', fontsize=13, fontweight='bold')
    ax.legend(title='Variante', title_fontsize=10, fontsize=9, loc='best', bbox_to_anchor=(1.05, 1))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    return ax


def plot_all_metrics_base_vs_robust_grid(
    base_results: pd.DataFrame,
    robust_results: pd.DataFrame,
    config: Config,
    save_path: Optional[str] = None
) -> None:
    """
    Crea una matriz de violin plots comparando base vs robust metric para todas las métricas,
    agrupado por dataset.
    """
    
    # Pares de métricas (base, robust)
    metric_pairs = [
        ("base_counterfactual_validity", "robust_counterfactual_validity"),
        ("base_counterfactual_proximityL1", "robust_counterfactual_proximityL1"),
        ("base_counterfactual_proximityL2", "robust_counterfactual_proximityL2"),
        ("base_counterfactual_plausibility", "robust_counterfactual_plausibility"),
        ("base_counterfactual_discriminative_power", "robust_counterfactual_discriminative_power")
    ]
    
    # Filtrar pares disponibles
    available_pairs = [
        (b, r) for b, r in metric_pairs 
        if b in base_results.columns and r in base_results.columns 
        and b in robust_results.columns and r in robust_results.columns
    ]
    
    if not available_pairs:
        print("Advertencia: No hay pares de métricas base/robust disponibles")
        return
    
    # Crear grid
    n_cols = 2
    n_rows = (len(available_pairs) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
    axes = axes.flatten()
    
    # Paleta de colores
    palette = {
        'Base (base metric)': '#7fbfff',
        'Base (robust metric)': '#1f77b4',
        'robx (base metric)': '#ffbf7f',
        'robx (robust metric)': '#ff7f0e'
    }
    hue_order = ['Base (base metric)', 'Base (robust metric)', 'robx (base metric)', 'robx (robust metric)']
    
    for idx, (base_metric, robust_metric) in enumerate(available_pairs):
        ax = axes[idx]
        
        # Preparar datos
        rows = []
        for _, row in base_results.iterrows():
            rows.append({'dataset_name': row['dataset_name'], 'value': row[base_metric], 'Variante': 'Base (base metric)'})
            rows.append({'dataset_name': row['dataset_name'], 'value': row[robust_metric], 'Variante': 'Base (robust metric)'})
        for _, row in robust_results.iterrows():
            rows.append({'dataset_name': row['dataset_name'], 'value': row[base_metric], 'Variante': f"{row['robustness_type']} (base metric)"})
            rows.append({'dataset_name': row['dataset_name'], 'value': row[robust_metric], 'Variante': f"{row['robustness_type']} (robust metric)"})
        
        combined = pd.DataFrame(rows)
        
        metric_name = config.METRIC_NAMES.get(base_metric, base_metric.replace('base_counterfactual_', ''))
        
        sns.violinplot(
            data=combined,
            x='dataset_name',
            y='value',
            hue='Variante',
            hue_order=hue_order,
            ax=ax,
            palette=palette,
            inner="box",
            cut=0
        )
        
        ax.set_xlabel('Dataset', fontsize=10, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
        ax.set_title(f'{metric_name} - Base vs Robust Metric', fontsize=11, fontweight='bold')
        ax.legend(title='Variante', fontsize=7, title_fontsize=8, loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Ocultar ejes no usados
    for idx in range(len(available_pairs), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Comparación Base vs Robust Metric: Base vs Robx por Dataset', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado: {save_path}")
    
    plt.close(fig)


def plot_all_metrics_grid_violin(
    base_results: pd.DataFrame,
    robust_results: pd.DataFrame,
    metrics: List[str],
    config: Config,
    save_path: Optional[str] = None
) -> None:
    """
    Crea una matriz de violin plots con todas las métricas.
    """
    
    # Filtrar métricas que existen
    available_metrics = [m for m in metrics if m in base_results.columns and m in robust_results.columns]
    
    if not available_metrics:
        print("Advertencia: No hay métricas disponibles")
        return
    
    # Preparar datos
    base_results_copy = base_results.copy()
    base_results_copy['Tipo'] = 'Base'
    
    robust_results_copy = robust_results.copy()
    robust_results_copy['Tipo'] = robust_results_copy['robustness_type']
    
    combined = pd.concat([base_results_copy, robust_results_copy], ignore_index=True)
    
    # Crear grid
    n_cols = 3
    n_rows = (len(available_metrics) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(config.FIGSIZE_MULTI[0], n_rows * 5))
    axes = axes.flatten()
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        metric_name = config.METRIC_NAMES.get(metric, metric.replace('base_counterfactual_', ''))
        
        sns.violinplot(
            data=combined,
            x='base_cf_method',
            y=metric,
            hue='Tipo',
            ax=ax,
            palette="Set2",
            inner="box",
            cut=0
        )
        
        ax.set_xlabel('Explainer Base', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
        ax.legend(title='Tipo', fontsize=9, title_fontsize=10, loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Ocultar ejes no usados
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Métricas Contrastive: Base vs Robustez Posthoc', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado: {save_path}")
    
    plt.show()


# ============================================================================
# FUNCIONES DE TESTS ESTADÍSTICOS
# ============================================================================

def perform_statistical_tests(
    base_results: pd.DataFrame,
    robust_results: pd.DataFrame,
    metrics: List[str],
    config: Config
) -> pd.DataFrame:
    """
    Realiza tests estadísticos para comparar métricas entre Base y Robust.
    Mann-Whitney U test para grupos independientes.
    """
    results = []
    
    available_metrics = [m for m in metrics if m in base_results.columns and m in robust_results.columns]
    
    for metric in available_metrics:
        metric_name = config.METRIC_NAMES.get(metric, metric)
        
        data_base = base_results[metric].dropna()
        data_robust = robust_results[metric].dropna()
        
        if len(data_base) < 2 or len(data_robust) < 2:
            continue
        
        try:
            mw_stat, mw_pvalue = stats.mannwhitneyu(
                data_base, data_robust, alternative='two-sided'
            )
        except Exception:
            mw_stat, mw_pvalue = np.nan, np.nan
        
        n1, n2 = len(data_base), len(data_robust)
        effect_size = 1 - (2 * mw_stat) / (n1 * n2) if n1 > 0 and n2 > 0 and not np.isnan(mw_stat) else np.nan
        
        # Determinar dirección del efecto
        mean_base = data_base.mean()
        mean_robust = data_robust.mean()
        if mean_base > mean_robust:
            direction = "Base > Robust"
            winner = "Base"
        elif mean_robust > mean_base:
            direction = "Robust > Base"
            winner = "Robust"
        else:
            direction = "Base = Robust"
            winner = "none"
        
        results.append({
            'metric': metric_name,
            'test': 'Mann-Whitney U',
            'comparison': 'Base vs Robust',
            'n_base': len(data_base),
            'n_robust': len(data_robust),
            'mean_base': round(mean_base, 4),
            'mean_robust': round(mean_robust, 4),
            'std_base': round(data_base.std(), 4),
            'std_robust': round(data_robust.std(), 4),
            'statistic': round(mw_stat, 4) if not np.isnan(mw_stat) else np.nan,
            'p_value': mw_pvalue,
            'effect_size': round(effect_size, 4) if not np.isnan(effect_size) else np.nan,
            'direction': direction,
            'winner': winner,
            'significant_0.05': mw_pvalue < 0.05 if not np.isnan(mw_pvalue) else False,
            'significant_0.01': mw_pvalue < 0.01 if not np.isnan(mw_pvalue) else False,
            'significant_0.001': mw_pvalue < 0.001 if not np.isnan(mw_pvalue) else False
        })
    
    return pd.DataFrame(results)


def perform_paired_statistical_tests(
    df: pd.DataFrame,
    base_metrics: List[str],
    robust_metrics: List[str],
    config: Config,
    group_name: str = "All"
) -> pd.DataFrame:
    """
    Realiza tests estadísticos pareados para comparar métricas base vs robust.
    Usa Wilcoxon signed-rank test.
    """
    results = []
    
    for base_metric, robust_metric in zip(base_metrics, robust_metrics):
        if base_metric not in df.columns or robust_metric not in df.columns:
            continue
        
        mask = df[base_metric].notna() & df[robust_metric].notna()
        data_base = df.loc[mask, base_metric]
        data_robust = df.loc[mask, robust_metric]
        
        if len(data_base) < 2:
            continue
        
        metric_name = config.METRIC_NAMES.get(base_metric, base_metric.replace('base_counterfactual_', ''))
        
        try:
            wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(
                data_base, data_robust, alternative='two-sided'
            )
        except Exception:
            wilcoxon_stat, wilcoxon_pvalue = np.nan, np.nan
        
        diff = data_robust - data_base
        n = len(diff[diff != 0])
        effect_size = 1 - (2 * wilcoxon_stat) / (n * (n + 1) / 2) if n > 0 and not np.isnan(wilcoxon_stat) else np.nan
        
        # Determinar dirección del efecto
        mean_base = data_base.mean()
        mean_robust = data_robust.mean()
        if mean_base > mean_robust:
            direction = "Base metric > Robust metric"
            winner = "Base metric"
        elif mean_robust > mean_base:
            direction = "Robust metric > Base metric"
            winner = "Robust metric"
        else:
            direction = "Base metric = Robust metric"
            winner = "none"
        
        results.append({
            'metric': metric_name,
            'test': 'Wilcoxon signed-rank',
            'comparison': f'{group_name}: Base vs Robust metric',
            'n_pairs': len(data_base),
            'mean_base': round(mean_base, 4),
            'mean_robust': round(mean_robust, 4),
            'mean_difference': round((data_robust - data_base).mean(), 4),
            'statistic': round(wilcoxon_stat, 4) if not np.isnan(wilcoxon_stat) else np.nan,
            'p_value': wilcoxon_pvalue,
            'effect_size': round(effect_size, 4) if not np.isnan(effect_size) else np.nan,
            'direction': direction,
            'winner': winner,
            'significant_0.05': wilcoxon_pvalue < 0.05 if not np.isnan(wilcoxon_pvalue) else False,
            'significant_0.01': wilcoxon_pvalue < 0.01 if not np.isnan(wilcoxon_pvalue) else False,
            'significant_0.001': wilcoxon_pvalue < 0.001 if not np.isnan(wilcoxon_pvalue) else False
        })
    
    return pd.DataFrame(results)


def perform_statistical_tests_by_dataset(
    base_results: pd.DataFrame,
    robust_results: pd.DataFrame,
    metrics: List[str],
    config: Config
) -> pd.DataFrame:
    """Realiza tests estadísticos por dataset."""
    results = []
    
    datasets = set(base_results['dataset_name'].unique()) & set(robust_results['dataset_name'].unique())
    available_metrics = [m for m in metrics if m in base_results.columns and m in robust_results.columns]
    
    for dataset in sorted(datasets):
        df_base = base_results[base_results['dataset_name'] == dataset]
        df_robust = robust_results[robust_results['dataset_name'] == dataset]
        
        for metric in available_metrics:
            metric_name = config.METRIC_NAMES.get(metric, metric)
            
            data_base = df_base[metric].dropna()
            data_robust = df_robust[metric].dropna()
            
            if len(data_base) < 2 or len(data_robust) < 2:
                continue
            
            try:
                mw_stat, mw_pvalue = stats.mannwhitneyu(
                    data_base, data_robust, alternative='two-sided'
                )
            except Exception:
                mw_stat, mw_pvalue = np.nan, np.nan
            
            n1, n2 = len(data_base), len(data_robust)
            effect_size = 1 - (2 * mw_stat) / (n1 * n2) if n1 > 0 and n2 > 0 and not np.isnan(mw_stat) else np.nan
            
            # Determinar dirección del efecto
            mean_base = data_base.mean()
            mean_robust = data_robust.mean()
            if mean_base > mean_robust:
                direction = "Base > Robust"
                winner = "Base"
            elif mean_robust > mean_base:
                direction = "Robust > Base"
                winner = "Robust"
            else:
                direction = "Base = Robust"
                winner = "none"
            
            results.append({
                'dataset': dataset,
                'metric': metric_name,
                'n_base': n1,
                'n_robust': n2,
                'mean_base': round(mean_base, 4),
                'mean_robust': round(mean_robust, 4),
                'p_value': mw_pvalue,
                'effect_size': round(effect_size, 4) if not np.isnan(effect_size) else np.nan,
                'direction': direction,
                'winner': winner,
                'significant_0.05': mw_pvalue < 0.05 if not np.isnan(mw_pvalue) else False
            })
    
    return pd.DataFrame(results)


def export_statistical_tests(
    base_results: pd.DataFrame,
    robust_results: pd.DataFrame,
    metrics: List[str],
    config: Config
) -> None:
    """Exporta todos los tests estadísticos a CSV."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Test Mann-Whitney global
    print("\n  Realizando tests Mann-Whitney U (Base vs Robust)...")
    global_tests = perform_statistical_tests(base_results, robust_results, metrics, config)
    if not global_tests.empty:
        global_tests.to_csv(os.path.join(config.OUTPUT_DIR, "statistical_tests_global.csv"), index=False)
        print(f"  ✓ Tests globales guardados")
        sig_count = global_tests['significant_0.05'].sum()
        total = len(global_tests)
        pct = (sig_count / total * 100) if total > 0 else 0
        print(f"  Métricas con diferencias significativas (p<0.05): {sig_count}/{total} ({pct:.1f}%)")
    
    # Tests por dataset
    print("\n  Realizando tests por dataset...")
    dataset_tests = perform_statistical_tests_by_dataset(base_results, robust_results, metrics, config)
    if not dataset_tests.empty:
        dataset_tests.to_csv(os.path.join(config.OUTPUT_DIR, "statistical_tests_by_dataset.csv"), index=False)
        print(f"  ✓ Tests por dataset guardados")
    
    # Tests pareados (base metric vs robust metric dentro de cada grupo)
    base_metrics = config.BASE_METRICS
    robust_metrics = config.ROBUST_METRICS
    
    print("\n  Realizando tests Wilcoxon pareados (Base metric vs Robust metric)...")
    paired_base = perform_paired_statistical_tests(base_results, base_metrics, robust_metrics, config, "Base")
    paired_robust = perform_paired_statistical_tests(robust_results, base_metrics, robust_metrics, config, "Robust")
    
    paired_combined = pd.concat([paired_base, paired_robust], ignore_index=True)
    if not paired_combined.empty:
        paired_combined.to_csv(os.path.join(config.OUTPUT_DIR, "statistical_tests_paired_base_vs_robust_metric.csv"), index=False)
        print(f"  ✓ Tests pareados guardados")
    
    # Resumen visual
    print("\n  === RESUMEN DE TESTS ESTADÍSTICOS ===")
    if not global_tests.empty:
        print("\n  Comparación Base vs Robust (Mann-Whitney U):")
        for _, row in global_tests.iterrows():
            sig = "***" if row['significant_0.001'] else ("**" if row['significant_0.01'] else ("*" if row['significant_0.05'] else ""))
            direction = f" [{row['direction']}]" if 'direction' in row and row['direction'] else ""
            winner = f" (mejor: {row['winner']})" if 'winner' in row and row['winner'] != 'none' and sig else ""
            print(f"    {row['metric']}: p={row['p_value']:.4e} {sig}{direction}{winner}")
    
    if not paired_combined.empty:
        print("\n  Comparación Base metric vs Robust metric (Wilcoxon signed-rank):")
        for _, row in paired_combined.iterrows():
            sig = "***" if row['significant_0.001'] else ("**" if row['significant_0.01'] else ("*" if row['significant_0.05'] else ""))
            direction = f" [{row['direction']}]" if 'direction' in row and row['direction'] else ""
            winner = f" (mejor: {row['winner']})" if 'winner' in row and row['winner'] != 'none' and sig else ""
            print(f"    {row['comparison']} - {row['metric']}: p={row['p_value']:.4e} {sig}{direction}{winner}")
    
    print("\n  Leyenda: * p<0.05, ** p<0.01, *** p<0.001")


# ============================================================================
# FUNCIONES DE EXPORTACIÓN
# ============================================================================

def export_aggregated_results(
    base_results: pd.DataFrame,
    robust_results: pd.DataFrame,
    metrics: List[str],
    config: Config
) -> None:
    """Exporta resultados agregados a CSV."""
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Filtrar métricas disponibles
    available_metrics = [m for m in metrics if m in base_results.columns and m in robust_results.columns]
    
    # Agregación por explainer base
    base_by_explainer = base_results.groupby('base_cf_method', observed=True)[available_metrics].mean().round(4)
    base_by_explainer['robustness_type'] = 'Base'
    base_by_explainer.to_csv(os.path.join(config.OUTPUT_DIR, "aggregated_by_explainer_base.csv"))
    
    robust_by_explainer = robust_results.groupby(['base_cf_method', 'robustness_type'], observed=True)[available_metrics].mean().round(4)
    robust_by_explainer.to_csv(os.path.join(config.OUTPUT_DIR, "aggregated_by_explainer_robust.csv"))
    
    # Agregación por dataset
    base_by_dataset = base_results.groupby('dataset_name', observed=True)[available_metrics].mean().round(4)
    base_by_dataset['robustness_type'] = 'Base'
    base_by_dataset.to_csv(os.path.join(config.OUTPUT_DIR, "aggregated_by_dataset_base.csv"))
    
    robust_by_dataset = robust_results.groupby(['dataset_name', 'robustness_type'], observed=True)[available_metrics].mean().round(4)
    robust_by_dataset.to_csv(os.path.join(config.OUTPUT_DIR, "aggregated_by_dataset_robust.csv"))
    
    # Accuracy agregado
    if 'accuracy' in base_results.columns:
        accuracy_base = base_results.groupby('dataset_name', observed=True)['accuracy'].mean().round(4)
        accuracy_base.to_csv(os.path.join(config.OUTPUT_DIR, "accuracy_by_dataset_base.csv"))
        
        accuracy_robust = robust_results.groupby(['dataset_name', 'robustness_type'], observed=True)['accuracy'].mean().round(4)
        accuracy_robust.to_csv(os.path.join(config.OUTPUT_DIR, "accuracy_by_dataset_robust.csv"))
    
    print(f"\nResultados agregados exportados a {config.OUTPUT_DIR}/")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal del script"""
    
    print("=" * 80)
    print("SCRIPT DE COMPARACIÓN CONTRASTIVE: Base vs Robustez Posthoc")
    print("=" * 80)
    
    config = Config()
    
    # 1. Encontrar y cargar archivos de resultados
    print("\n[PASO 1] Cargando resultados de Contrastive...")
    print("-" * 80)
    
    result_files = find_result_files(config)
    df_raw = load_contrastive_results(result_files, config)
    
    if df_raw.empty:
        print("ERROR: No se pudieron cargar los datos. Abortando.")
        return
    
    # 2. Procesar datos
    print("\n[PASO 2] Procesando datos...")
    print("-" * 80)
    
    # Calcular accuracy
    df_raw = calculate_accuracy(df_raw)
    
    # Separar por robustez
    base_results, robust_results = separate_by_robustness(df_raw)
    
    # Definir métricas a usar (base y robustas, SIN accuracy para violin)
    all_metrics = config.BASE_METRICS + config.ROBUST_METRICS
    
    # Filtrar solo métricas disponibles (accuracy se visualiza con barras, no violin)
    available_metrics = [m for m in all_metrics if m in df_raw.columns]
    print(f"\nMétricas disponibles para comparar: {len(available_metrics)}")
    for m in available_metrics:
        print(f"  - {config.METRIC_NAMES.get(m, m)}")
    
    print(f"\nDatos Base: {len(base_results)} registros")
    print(f"Datos Robust: {len(robust_results)} registros")
    print(f"Explainers base: {base_results['base_cf_method'].unique()}")
    print(f"Datasets: {base_results['dataset_name'].unique()}")
    print(f"Tipos de robustez: {robust_results['robustness_type'].unique()}")
    
    # 3. Crear visualizaciones
    print("\n[PASO 3] Generando visualizaciones...")
    print("-" * 80)
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Gráfico de barras de accuracy por dataset (PRINCIPAL para accuracy)
    print("Generando gráfico de BARRAS de accuracy por dataset...")
    if 'accuracy' in base_results.columns:
        try:
            fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
            # Calcular accuracy promedio por dataset
            acc_base = base_results.groupby('dataset_name', observed=True)['accuracy'].mean()
            acc_robust = robust_results.groupby('dataset_name', observed=True)['accuracy'].mean()
            
            datasets = sorted(set(acc_base.index) | set(acc_robust.index))
            x = np.arange(len(datasets))
            width = 0.35
            
            ax.bar(x - width/2, [acc_base.get(d, 0) for d in datasets], width, label='Base', color='#1f77b4')
            ax.bar(x + width/2, [acc_robust.get(d, 0) for d in datasets], width, label='Robust', color='#ff7f0e')
            
            ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
            ax.set_title('Accuracy por Dataset: Base vs Robust', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(datasets, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            save_path = os.path.join(config.OUTPUT_DIR, "01_accuracy_by_dataset.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ {save_path}")
        except Exception as e:
            print(f"✗ Error generando gráfico de accuracy: {e}")
    
    # Gráficos individuales por métrica (agrupados por explainer base) - SIN accuracy
    print("Generando violin plots por explainer...")
    for metric in available_metrics:
        try:
            fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
            plot_metric_violin_by_explainer(base_results, robust_results, metric, config, ax)
            metric_name = config.METRIC_NAMES.get(metric, metric).replace(' ', '_').replace('(', '').replace(')', '')
            save_path = os.path.join(config.OUTPUT_DIR, f"02_violin_explainer_{metric_name}.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ {save_path}")
        except Exception as e:
            print(f"✗ Error generando gráfico para {metric}: {e}")
    
    # Gráficos individuales por métrica (agrupados por dataset) - SIN accuracy
    print("\nGenerando violin plots por dataset (métricas, sin accuracy)...")
    for metric in available_metrics:
        try:
            fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
            plot_metric_violin_by_dataset(base_results, robust_results, metric, config, ax)
            metric_name = config.METRIC_NAMES.get(metric, metric).replace(' ', '_').replace('(', '').replace(')', '')
            save_path = os.path.join(config.OUTPUT_DIR, f"03_violin_dataset_{metric_name}.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ {save_path}")
        except Exception as e:
            print(f"✗ Error generando gráfico para {metric}: {e}")
    
    # Grid de todos los gráficos (métricas, sin accuracy)
    print("\nGenerando grid de violin plots (métricas)...")
    try:
        fig = plt.figure(figsize=config.FIGSIZE_MULTI)
        plot_all_metrics_grid_violin(base_results, robust_results, available_metrics, config,
                                     os.path.join(config.OUTPUT_DIR, "04_violin_grid.png"))
        plt.close(fig)
    except Exception as e:
        print(f"✗ Error generando grid: {e}")
    
    # Gráficos comparando base metric vs robust metric por dataset (4 violines por dataset)
    print("\nGenerando violin plots Base Metric vs Robust Metric por dataset...")
    metric_pairs = [
        ("base_counterfactual_validity", "robust_counterfactual_validity"),
        ("base_counterfactual_proximityL1", "robust_counterfactual_proximityL1"),
        ("base_counterfactual_proximityL2", "robust_counterfactual_proximityL2"),
        ("base_counterfactual_plausibility", "robust_counterfactual_plausibility"),
        ("base_counterfactual_discriminative_power", "robust_counterfactual_discriminative_power")
    ]
    for base_metric, robust_metric in metric_pairs:
        try:
            fig, ax = plt.subplots(figsize=(14, 7))
            plot_metric_base_vs_robust_by_dataset(base_results, robust_results, base_metric, robust_metric, config, ax)
            metric_name = config.METRIC_NAMES.get(base_metric, base_metric.replace('base_counterfactual_', '')).replace(' ', '_')
            save_path = os.path.join(config.OUTPUT_DIR, f"05_violin_base_vs_robust_{metric_name}.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ {save_path}")
        except Exception as e:
            print(f"✗ Error generando gráfico base vs robust para {base_metric}: {e}")
    
    # Grid de base vs robust
    print("\nGenerando grid de violin plots Base Metric vs Robust Metric...")
    try:
        plot_all_metrics_base_vs_robust_grid(base_results, robust_results, config,
                                              os.path.join(config.OUTPUT_DIR, "06_violin_base_vs_robust_grid.png"))
    except Exception as e:
        print(f"✗ Error generando grid base vs robust: {e}")
    
    # 4. Exportar resultados
    print("\n[PASO 4] Exportando resultados agregados...")
    print("-" * 80)
    export_aggregated_results(base_results, robust_results, available_metrics, config)
    
    # 5. Tests estadísticos
    print("\n[PASO 5] Realizando tests estadísticos...")
    print("-" * 80)
    export_statistical_tests(base_results, robust_results, available_metrics, config)
    
    print("\n" + "=" * 80)
    print(f"✓ COMPLETADO. Resultados en: {config.OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
