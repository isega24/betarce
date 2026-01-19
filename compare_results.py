"""
Script para comparar métricas entre experimentos "all" (config_exp.yaml) y "contrastive" (config_exp_contrastive.yaml)

Este script:
1. Carga resultados de múltiples runs
2. Agrega las métricas por dataset, clasificador y explainer base
3. Genera gráficos de barras comparativos para cada métrica
4. Exporta resultados agregados a CSV

Estructura de datos:
- Cada fila = un counterfactual example
- Métricas con prefijo "base_counterfactual_" (gs, dice, face)
- Métricas con prefijo "robust_counterfactual_" (roar, rbr, betarob, robx)
- dataset_name: fico, wine_quality, etc.
- model_type_to_use: lightgbm, neural_network, etc.
- base_cf_method: gs, dice, face, etc.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend no-interactivo para sbatch
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
    
    # Directorios de experimentos
    EXPERIMENT_DIRS = {
        "all": "trial/results",  # Config: config_exp.yaml (se guarda en trial/)
        "contrastive": "contrastive_trial/results"  # Config: config_exp_contrastive.yaml
    }
    
    # Clasificadores por experimento
    CLASSIFIERS = {
        "all": ["neural_network", "lightgbm", "logistic_regression"],
        "contrastive": ["neural_network_contrastive"]
    }
    
    # Explainers base (generan counterfactuals base)
    BASE_EXPLAINERS = ["gs", "dice", "face"]
    
    # Explainers robustos (generan counterfactuals robustos)
    ROBUST_EXPLAINERS = ["roar", "rbr"]
    
    # Explainers posthoc (aplican robustez a base)
    POSTHOC_EXPLAINERS = ["betarob", "robx"]
    
    # Métricas a comparar (disponibles con prefijos)
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
        "robust_counterfactual_discriminative_power": "dpow (robust)"
    }
    
    # Datasets (definidos en config YAML)
    DATASETS = [
        "fico",
        "wine_quality",
        "breast_cancer",
        "car_eval",
        "rice",
        "diabetes"
    ]
    
    # Estilos de visualización
    COLORS = {
        "all": "#1f77b4",
        "contrastive": "#ff7f0e"
    }
    
    # Directorio de salida
    OUTPUT_DIR = "results_comparison"
    
    # Figsize para gráficos
    FIGSIZE_SINGLE = (12, 6)
    FIGSIZE_MULTI = (16, 10)


# ============================================================================
# FUNCIONES DE CARGA Y PROCESAMIENTO
# ============================================================================

def find_result_files(exp_type: str, config: Config) -> List[str]:
    """
    Encuentra todos los archivos de resultados para un tipo de experimento.
    
    Args:
        exp_type: "all" o "contrastive"
        config: Objeto Config con rutas
        
    Returns:
        Lista de rutas a archivos CSV
    """
    base_path = os.path.join(config.RESULTS_BASE, config.EXPERIMENT_DIRS[exp_type])
    
    # Buscar archivos CSV recursivamente
    result_files = glob.glob(
        os.path.join(base_path, "**", "results", "*.csv"),
        recursive=True
    )
    
    print(f"[{exp_type.upper()}] Encontrados {len(result_files)} archivos de resultados")
    
    return result_files


def load_and_aggregate_results(
    result_files: List[str],
    exp_type: str,
    config: Config
) -> pd.DataFrame:
    """
    Carga y agrega resultados de múltiples runs.
    
    Args:
        result_files: Lista de rutas a archivos CSV
        exp_type: "all" o "contrastive"
        config: Objeto Config
        
    Returns:
        DataFrame con resultados agregados
    """
    dfs = []
    
    for file_path in result_files:
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"Advertencia: No se pudo cargar {file_path}: {e}")
            continue
    
    if not dfs:
        print(f"Advertencia: No hay datos para {exp_type}")
        return pd.DataFrame()
    
    # Concatenar todos los DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['experiment_type'] = exp_type
    
    print(f"[{exp_type.upper()}] Datos cargados: {len(combined_df)} filas")
    
    return combined_df


def prepare_metrics_for_analysis(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Prepara los datos para análisis renombrando y normalizando columnas de métricas.
    
    Args:
        df: DataFrame crudo
        config: Objeto Config
        
    Returns:
        DataFrame preparado con métricas agregadas
    """
    
    df_copy = df.copy()
    
    # Renombrar métrica de discriminative power
    if 'base_counterfactual_discriminative_power' in df_copy.columns:
        df_copy['base_counterfactual_dpow'] = df_copy['base_counterfactual_discriminative_power']
    
    if 'robust_counterfactual_discriminative_power' in df_copy.columns:
        df_copy['robust_counterfactual_dpow'] = df_copy['robust_counterfactual_discriminative_power']
    
    return df_copy


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN (Violin Plots con Seaborn)
# ============================================================================

def plot_metric_violin_by_explainer(
    all_results: pd.DataFrame,
    contrastive_results: pd.DataFrame,
    metric: str,
    config: Config,
    ax: Optional[plt.Axes] = None
) -> Optional[plt.Axes]:
    """
    Crea violin plot comparativo de una métrica por explainer base y por clasificador (en "all").
    
    Args:
        all_results: Resultados de "all" (separados por clasificador)
        contrastive_results: Resultados de "contrastive"
        metric: Nombre de la columna de métrica
        config: Objeto Config
        ax: Axes para dibujar
        
    Returns:
        Axes del gráfico
    """
    
    # Preparar datos: agregar etiqueta de experimento y clasificador
    all_results_copy = all_results.copy()
    all_results_copy['Experimento'] = all_results_copy['model_type_to_use'].apply(
        lambda x: f"all-{x}"
    )
    
    contrastive_results_copy = contrastive_results.copy()
    contrastive_results_copy['Experimento'] = 'contrastive'
    
    # Combinar datos
    combined = pd.concat([all_results_copy, contrastive_results_copy], ignore_index=True)
    
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
        hue='Experimento',
        ax=ax,
        palette="Set2",
        inner="box",
        split=False,
        cut=0
    )
    
    # Configurar ejes
    ax.set_xlabel('Explainer Base', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} - Comparación por Clasificador', fontsize=13, fontweight='bold')
    ax.legend(title='Experimento', title_fontsize=11, fontsize=10, loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return ax


def plot_metric_violin_by_dataset(
    all_results: pd.DataFrame,
    contrastive_results: pd.DataFrame,
    metric: str,
    config: Config,
    ax: Optional[plt.Axes] = None
) -> Optional[plt.Axes]:
    """
    Crea violin plot comparativo de una métrica por dataset.
    
    Args:
        all_results: Resultados de "all"
        contrastive_results: Resultados de "contrastive"
        metric: Nombre de la columna de métrica
        config: Objeto Config
        ax: Axes para dibujar
        
    Returns:
        Axes del gráfico
    """
    
    # Preparar datos
    all_results_copy = all_results.copy()
    all_results_copy['Experimento'] = all_results_copy['model_type_to_use'].apply(
        lambda x: f"all-{x}"
    )
    
    contrastive_results_copy = contrastive_results.copy()
    contrastive_results_copy['Experimento'] = 'contrastive'
    
    # Combinar datos
    combined = pd.concat([all_results_copy, contrastive_results_copy], ignore_index=True)
    
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
        hue='Experimento',
        ax=ax,
        palette="Set2",
        inner="box",
        cut=0
    )
    
    # Configurar ejes
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} - Comparación por Dataset', fontsize=13, fontweight='bold')
    ax.legend(title='Experimento', title_fontsize=11, fontsize=10, loc='best', bbox_to_anchor=(1.05, 1))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return ax


def plot_metric_base_vs_robust_by_dataset(
    all_results: pd.DataFrame,
    contrastive_results: pd.DataFrame,
    base_metric: str,
    robust_metric: str,
    config: Config,
    ax: Optional[plt.Axes] = None
) -> Optional[plt.Axes]:
    """
    Crea violin plot comparativo de una métrica base vs su versión robust,
    para ambos experimentos (all y contrastive) agrupado por dataset.
    
    Muestra 4 violines por dataset:
    - all (base)
    - all (robust)
    - contrastive (base)
    - contrastive (robust)
    """
    
    # Verificar que las métricas existen
    if base_metric not in all_results.columns or robust_metric not in all_results.columns:
        print(f"Advertencia: Métricas {base_metric} o {robust_metric} no encontradas en all_results")
        return None
    if base_metric not in contrastive_results.columns or robust_metric not in contrastive_results.columns:
        print(f"Advertencia: Métricas {base_metric} o {robust_metric} no encontradas en contrastive_results")
        return None
    
    # Preparar datos - transformar a formato largo para comparar base vs robust
    rows = []
    
    # all - base
    for _, row in all_results.iterrows():
        rows.append({
            'dataset_name': row['dataset_name'],
            'value': row[base_metric],
            'Variante': 'all (base)'
        })
    
    # all - robust
    for _, row in all_results.iterrows():
        rows.append({
            'dataset_name': row['dataset_name'],
            'value': row[robust_metric],
            'Variante': 'all (robust)'
        })
    
    # contrastive - base
    for _, row in contrastive_results.iterrows():
        rows.append({
            'dataset_name': row['dataset_name'],
            'value': row[base_metric],
            'Variante': 'contrastive (base)'
        })
    
    # contrastive - robust
    for _, row in contrastive_results.iterrows():
        rows.append({
            'dataset_name': row['dataset_name'],
            'value': row[robust_metric],
            'Variante': 'contrastive (robust)'
        })
    
    combined = pd.DataFrame(rows)
    
    # Crear gráfico si no existe
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))
    
    # Nombre limpio de la métrica (usar el nombre base sin prefijo)
    metric_name = config.METRIC_NAMES.get(base_metric, base_metric.replace('base_counterfactual_', ''))
    
    # Paleta de colores: azul para all, naranja para contrastive, más claro para base, más oscuro para robust
    palette = {
        'all (base)': '#7fbfff',           # azul claro
        'all (robust)': '#1f77b4',         # azul oscuro
        'contrastive (base)': '#ffbf7f',   # naranja claro
        'contrastive (robust)': '#ff7f0e'  # naranja oscuro
    }
    
    # Orden para las variantes
    hue_order = ['all (base)', 'all (robust)', 'contrastive (base)', 'contrastive (robust)']
    
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
    ax.set_title(f'{metric_name} - Base vs Robust por Experimento y Dataset', fontsize=13, fontweight='bold')
    ax.legend(title='Variante', title_fontsize=10, fontsize=9, loc='best', bbox_to_anchor=(1.05, 1))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    return ax


def plot_all_metrics_base_vs_robust_grid(
    all_results: pd.DataFrame,
    contrastive_results: pd.DataFrame,
    config: Config,
    save_path: Optional[str] = None
) -> None:
    """
    Crea una matriz de violin plots comparando base vs robust para todas las métricas,
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
        if b in all_results.columns and r in all_results.columns 
        and b in contrastive_results.columns and r in contrastive_results.columns
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
        'all (base)': '#7fbfff',
        'all (robust)': '#1f77b4',
        'contrastive (base)': '#ffbf7f',
        'contrastive (robust)': '#ff7f0e'
    }
    hue_order = ['all (base)', 'all (robust)', 'contrastive (base)', 'contrastive (robust)']
    
    for idx, (base_metric, robust_metric) in enumerate(available_pairs):
        ax = axes[idx]
        
        # Preparar datos
        rows = []
        for _, row in all_results.iterrows():
            rows.append({'dataset_name': row['dataset_name'], 'value': row[base_metric], 'Variante': 'all (base)'})
            rows.append({'dataset_name': row['dataset_name'], 'value': row[robust_metric], 'Variante': 'all (robust)'})
        for _, row in contrastive_results.iterrows():
            rows.append({'dataset_name': row['dataset_name'], 'value': row[base_metric], 'Variante': 'contrastive (base)'})
            rows.append({'dataset_name': row['dataset_name'], 'value': row[robust_metric], 'Variante': 'contrastive (robust)'})
        
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
        ax.set_title(f'{metric_name} - Base vs Robust', fontsize=11, fontweight='bold')
        ax.legend(title='Variante', fontsize=7, title_fontsize=8, loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Ocultar ejes no usados
    for idx in range(len(available_pairs), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Comparación Base vs Robust: all vs contrastive por Dataset', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado: {save_path}")
    
    plt.close(fig)


def plot_all_metrics_grid_violin(
    all_results: pd.DataFrame,
    contrastive_results: pd.DataFrame,
    metrics: List[str],
    config: Config,
    save_path: Optional[str] = None
) -> None:
    """
    Crea una matriz de violin plots con todas las métricas (comparación por explainer).
    
    Args:
        all_results: Resultados de "all"
        contrastive_results: Resultados de "contrastive"
        metrics: Lista de métricas
        config: Objeto Config
        save_path: Ruta para guardar
    """
    
    # Filtrar métricas que existen
    available_metrics = [m for m in metrics if m in all_results.columns and m in contrastive_results.columns]
    
    if not available_metrics:
        print("Advertencia: No hay métricas disponibles")
        return
    
    # Preparar datos
    all_results_copy = all_results.copy()
    all_results_copy['Experimento'] = all_results_copy['model_type_to_use'].apply(
        lambda x: f"all-{x}"
    )
    
    contrastive_results_copy = contrastive_results.copy()
    contrastive_results_copy['Experimento'] = 'contrastive'
    
    combined = pd.concat([all_results_copy, contrastive_results_copy], ignore_index=True)
    
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
            hue='Experimento',
            ax=ax,
            palette="Set2",
            inner="box",
            cut=0
        )
        
        ax.set_xlabel('Explainer Base', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
        ax.legend(title='Experimento', fontsize=9, title_fontsize=10, loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Ocultar ejes no usados
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Distribución de Métricas: all (por clasificador) vs contrastive', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado: {save_path}")
    
    plt.close(fig)



# ============================================================================
# FUNCIONES DE TESTS ESTADÍSTICOS
# ============================================================================

def perform_statistical_tests(
    all_results: pd.DataFrame,
    contrastive_results: pd.DataFrame,
    metrics: List[str],
    config: Config
) -> pd.DataFrame:
    """
    Realiza tests estadísticos para comparar métricas entre grupos.
    
    Tests utilizados:
    - Mann-Whitney U: Para comparar grupos independientes (all vs contrastive)
    """
    results = []
    
    available_metrics = [m for m in metrics if m in all_results.columns and m in contrastive_results.columns]
    
    for metric in available_metrics:
        metric_name = config.METRIC_NAMES.get(metric, metric)
        
        data_all = all_results[metric].dropna()
        data_contrastive = contrastive_results[metric].dropna()
        
        if len(data_all) < 2 or len(data_contrastive) < 2:
            continue
        
        # Mann-Whitney U test
        try:
            mw_stat, mw_pvalue = stats.mannwhitneyu(
                data_all, data_contrastive, alternative='two-sided'
            )
        except Exception:
            mw_stat, mw_pvalue = np.nan, np.nan
        
        n1, n2 = len(data_all), len(data_contrastive)
        effect_size = 1 - (2 * mw_stat) / (n1 * n2) if n1 > 0 and n2 > 0 and not np.isnan(mw_stat) else np.nan
        
        # Determinar dirección del efecto
        mean_a = data_all.mean()
        mean_c = data_contrastive.mean()
        if mw_pvalue < 0.05 if not np.isnan(mw_pvalue) else False:
            if mean_c > mean_a:
                direction = 'contrastive > all'
                winner = 'contrastive'
            else:
                direction = 'all > contrastive'
                winner = 'all'
        else:
            direction = 'No significant difference'
            winner = 'None'
        
        results.append({
            'metric': metric_name,
            'test': 'Mann-Whitney U',
            'comparison': 'all vs contrastive',
            'n_all': len(data_all),
            'n_contrastive': len(data_contrastive),
            'mean_all': round(mean_a, 4),
            'mean_contrastive': round(mean_c, 4),
            'std_all': round(data_all.std(), 4),
            'std_contrastive': round(data_contrastive.std(), 4),
            'statistic': round(mw_stat, 4) if not np.isnan(mw_stat) else np.nan,
            'p_value': mw_pvalue,
            'effect_size': round(effect_size, 4) if not np.isnan(effect_size) else np.nan,
            'significant_0.05': mw_pvalue < 0.05 if not np.isnan(mw_pvalue) else False,
            'significant_0.01': mw_pvalue < 0.01 if not np.isnan(mw_pvalue) else False,
            'significant_0.001': mw_pvalue < 0.001 if not np.isnan(mw_pvalue) else False,
            'direction': direction,
            'winner': winner
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
        mean_b = data_base.mean()
        mean_r = data_robust.mean()
        if wilcoxon_pvalue < 0.05 if not np.isnan(wilcoxon_pvalue) else False:
            if mean_r > mean_b:
                direction = 'Robust > Base'
                winner = 'Robust'
            else:
                direction = 'Base > Robust'
                winner = 'Base'
        else:
            direction = 'No significant difference'
            winner = 'None'
        
        results.append({
            'metric': metric_name,
            'test': 'Wilcoxon signed-rank',
            'comparison': f'{group_name}: Base vs Robust',
            'n_pairs': len(data_base),
            'mean_base': round(mean_b, 4),
            'mean_robust': round(mean_r, 4),
            'mean_difference': round((data_robust - data_base).mean(), 4),
            'statistic': round(wilcoxon_stat, 4) if not np.isnan(wilcoxon_stat) else np.nan,
            'p_value': wilcoxon_pvalue,
            'effect_size': round(effect_size, 4) if not np.isnan(effect_size) else np.nan,
            'significant_0.05': wilcoxon_pvalue < 0.05 if not np.isnan(wilcoxon_pvalue) else False,
            'significant_0.01': wilcoxon_pvalue < 0.01 if not np.isnan(wilcoxon_pvalue) else False,
            'significant_0.001': wilcoxon_pvalue < 0.001 if not np.isnan(wilcoxon_pvalue) else False,
            'direction': direction,
            'winner': winner
        })
    
    return pd.DataFrame(results)


def perform_statistical_tests_by_dataset(
    all_results: pd.DataFrame,
    contrastive_results: pd.DataFrame,
    metrics: List[str],
    config: Config
) -> pd.DataFrame:
    """Realiza tests estadísticos por dataset."""
    results = []
    
    datasets = set(all_results['dataset_name'].unique()) & set(contrastive_results['dataset_name'].unique())
    available_metrics = [m for m in metrics if m in all_results.columns and m in contrastive_results.columns]
    
    for dataset in sorted(datasets):
        df_all = all_results[all_results['dataset_name'] == dataset]
        df_con = contrastive_results[contrastive_results['dataset_name'] == dataset]
        
        for metric in available_metrics:
            metric_name = config.METRIC_NAMES.get(metric, metric)
            
            data_all = df_all[metric].dropna()
            data_contrastive = df_con[metric].dropna()
            
            if len(data_all) < 2 or len(data_contrastive) < 2:
                continue
            
            try:
                mw_stat, mw_pvalue = stats.mannwhitneyu(
                    data_all, data_contrastive, alternative='two-sided'
                )
            except Exception:
                mw_stat, mw_pvalue = np.nan, np.nan
            
            n1, n2 = len(data_all), len(data_contrastive)
            effect_size = 1 - (2 * mw_stat) / (n1 * n2) if n1 > 0 and n2 > 0 and not np.isnan(mw_stat) else np.nan
            
            # Determinar dirección del efecto
            mean_a = data_all.mean()
            mean_c = data_contrastive.mean()
            if mw_pvalue < 0.05 if not np.isnan(mw_pvalue) else False:
                if mean_c > mean_a:
                    direction = 'contrastive > all'
                    winner = 'contrastive'
                else:
                    direction = 'all > contrastive'
                    winner = 'all'
            else:
                direction = 'No significant difference'
                winner = 'None'
            
            results.append({
                'dataset': dataset,
                'metric': metric_name,
                'n_all': n1,
                'n_contrastive': n2,
                'mean_all': round(mean_a, 4),
                'mean_contrastive': round(mean_c, 4),
                'p_value': mw_pvalue,
                'effect_size': round(effect_size, 4) if not np.isnan(effect_size) else np.nan,
                'significant_0.05': mw_pvalue < 0.05 if not np.isnan(mw_pvalue) else False,
                'direction': direction,
                'winner': winner
            })
    
    return pd.DataFrame(results)


def export_statistical_tests(
    all_results: pd.DataFrame,
    contrastive_results: pd.DataFrame,
    metrics: List[str],
    config: Config
) -> None:
    """Exporta todos los tests estadísticos a CSV."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Test Mann-Whitney global
    print("\n  Realizando tests Mann-Whitney U (all vs contrastive)...")
    global_tests = perform_statistical_tests(all_results, contrastive_results, metrics, config)
    if not global_tests.empty:
        global_tests.to_csv(os.path.join(config.OUTPUT_DIR, "statistical_tests_global.csv"), index=False)
        print(f"  ✓ Tests globales guardados")
        sig_count = global_tests['significant_0.05'].sum()
        total = len(global_tests)
        pct = (sig_count / total * 100) if total > 0 else 0
        print(f"  Métricas con diferencias significativas (p<0.05): {sig_count}/{total} ({pct:.1f}%)")
    
    # Tests por dataset
    print("\n  Realizando tests por dataset...")
    dataset_tests = perform_statistical_tests_by_dataset(all_results, contrastive_results, metrics, config)
    if not dataset_tests.empty:
        dataset_tests.to_csv(os.path.join(config.OUTPUT_DIR, "statistical_tests_by_dataset.csv"), index=False)
        print(f"  ✓ Tests por dataset guardados")
    
    # Tests pareados (base vs robust)
    base_metrics = config.BASE_METRICS
    robust_metrics = config.ROBUST_METRICS
    
    print("\n  Realizando tests Wilcoxon pareados (Base vs Robust)...")
    paired_all = perform_paired_statistical_tests(all_results, base_metrics, robust_metrics, config, "all")
    paired_contrastive = perform_paired_statistical_tests(contrastive_results, base_metrics, robust_metrics, config, "contrastive")
    
    paired_combined = pd.concat([paired_all, paired_contrastive], ignore_index=True)
    if not paired_combined.empty:
        paired_combined.to_csv(os.path.join(config.OUTPUT_DIR, "statistical_tests_paired_base_vs_robust.csv"), index=False)
        print(f"  ✓ Tests pareados guardados")
    
    # Resumen visual
    print("\n  === RESUMEN DE TESTS ESTADÍSTICOS ===")
    if not global_tests.empty:
        print("\n  Comparación all vs contrastive (Mann-Whitney U):")
        for _, row in global_tests.iterrows():
            sig = "***" if row['significant_0.001'] else ("**" if row['significant_0.01'] else ("*" if row['significant_0.05'] else ""))
            direction = f" [{row['direction']}]" if 'direction' in row and row['direction'] else ""
            winner = f" (mejor: {row['winner']})" if 'winner' in row and row['winner'] != 'none' and sig else ""
            print(f"    {row['metric']}: p={row['p_value']:.4e} {sig}{direction}{winner}")
    
    if not paired_combined.empty:
        print("\n  Comparación Base vs Robust (Wilcoxon signed-rank):")
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
    all_results: pd.DataFrame,
    contrastive_results: pd.DataFrame,
    metrics: List[str],
    config: Config
) -> None:
    """
    Exporta resultados agregados a CSV.
    
    Args:
        all_results: Resultados de "all"
        contrastive_results: Resultados de "contrastive"
        metrics: Lista de métricas
        config: Objeto Config
    """
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Filtrar métricas disponibles
    available_metrics = [m for m in metrics if m in all_results.columns and m in contrastive_results.columns]
    
    # Agregación por explainer base y métrica
    all_by_explainer = all_results.groupby('base_cf_method', observed=True)[available_metrics].mean().round(4)
    all_by_explainer['experiment'] = 'all'
    all_by_explainer.to_csv(os.path.join(config.OUTPUT_DIR, "aggregated_by_explainer_all.csv"))
    
    contrastive_by_explainer = contrastive_results.groupby('base_cf_method', observed=True)[available_metrics].mean().round(4)
    contrastive_by_explainer['experiment'] = 'contrastive'
    contrastive_by_explainer.to_csv(os.path.join(config.OUTPUT_DIR, "aggregated_by_explainer_contrastive.csv"))
    
    # Agregación por dataset y métrica
    all_by_dataset = all_results.groupby('dataset_name', observed=True)[available_metrics].mean().round(4)
    all_by_dataset['experiment'] = 'all'
    all_by_dataset.to_csv(os.path.join(config.OUTPUT_DIR, "aggregated_by_dataset_all.csv"))
    
    contrastive_by_dataset = contrastive_results.groupby('dataset_name', observed=True)[available_metrics].mean().round(4)
    contrastive_by_dataset['experiment'] = 'contrastive'
    contrastive_by_dataset.to_csv(os.path.join(config.OUTPUT_DIR, "aggregated_by_dataset_contrastive.csv"))
    
    # Agregación por clasificador
    all_by_classifier = all_results.groupby('model_type_to_use', observed=True)[available_metrics].mean().round(4)
    all_by_classifier['experiment'] = 'all'
    all_by_classifier.to_csv(os.path.join(config.OUTPUT_DIR, "aggregated_by_classifier_all.csv"))
    
    contrastive_by_classifier = contrastive_results.groupby('model_type_to_use', observed=True)[available_metrics].mean().round(4)
    contrastive_by_classifier['experiment'] = 'contrastive'
    contrastive_by_classifier.to_csv(os.path.join(config.OUTPUT_DIR, "aggregated_by_classifier_contrastive.csv"))
    
    print(f"\nResultados agregados exportados a {config.OUTPUT_DIR}/")



# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal del script"""
    
    print("=" * 80)
    print("SCRIPT DE COMPARACIÓN DE RESULTADOS: all vs contrastive")
    print("=" * 80)
    
    config = Config()
    
    # 1. Encontrar y cargar archivos de resultados
    print("\n[PASO 1] Cargando resultados...")
    print("-" * 80)
    
    all_files = find_result_files("all", config)
    contrastive_files = find_result_files("contrastive", config)
    
    all_raw = load_and_aggregate_results(all_files, "all", config)
    contrastive_raw = load_and_aggregate_results(contrastive_files, "contrastive", config)
    
    if all_raw.empty or contrastive_raw.empty:
        print("ERROR: No se pudieron cargar los datos. Abortando.")
        return
    
    # 2. Procesar datos
    print("\n[PASO 2] Procesando datos...")
    print("-" * 80)
    
    # Preparar métricas
    all_raw = prepare_metrics_for_analysis(all_raw, config)
    contrastive_raw = prepare_metrics_for_analysis(contrastive_raw, config)
    
    # Definir métricas a usar (base y robustas)
    all_metrics = config.BASE_METRICS + config.ROBUST_METRICS
    
    # Filtrar solo métricas disponibles
    available_metrics = [m for m in all_metrics if m in all_raw.columns and m in contrastive_raw.columns]
    print(f"Métricas disponibles para comparar: {len(available_metrics)}")
    for m in available_metrics:
        print(f"  - {config.METRIC_NAMES.get(m, m)}")
    
    print(f"\nDatos 'all': {len(all_raw)} registros")
    print(f"Datos 'contrastive': {len(contrastive_raw)} registros")
    print(f"Explainers base (all): {all_raw['base_cf_method'].unique()}")
    print(f"Explainers base (contrastive): {contrastive_raw['base_cf_method'].unique()}")
    print(f"Clasificadores (all): {all_raw['model_type_to_use'].unique()}")
    print(f"Clasificadores (contrastive): {contrastive_raw['model_type_to_use'].unique()}")
    
    # 3. Crear visualizaciones
    print("\n[PASO 3] Generando visualizaciones (Violin Plots para métricas)...")
    print("-" * 80)
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Gráficos individuales por métrica (agrupados por explainer base)
    print("Generando violin plots por explainer (métricas)...")
    for metric in available_metrics:
        try:
            fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
            plot_metric_violin_by_explainer(all_raw, contrastive_raw, metric, config, ax)
            metric_name = config.METRIC_NAMES.get(metric, metric)
            save_path = os.path.join(config.OUTPUT_DIR, f"01_violin_explainer_{metric_name.replace(' ', '_')}.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ {save_path}")
        except Exception as e:
            print(f"✗ Error generando gráfico para {metric}: {e}")
    
    # Gráficos individuales por métrica (agrupados por dataset)
    print("\nGenerando violin plots por dataset (métricas)...")
    for metric in available_metrics:
        try:
            fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
            plot_metric_violin_by_dataset(all_raw, contrastive_raw, metric, config, ax)
            metric_name = config.METRIC_NAMES.get(metric, metric)
            save_path = os.path.join(config.OUTPUT_DIR, f"02_violin_dataset_{metric_name.replace(' ', '_')}.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ {save_path}")
        except Exception as e:
            print(f"✗ Error generando gráfico para {metric}: {e}")
    
    # Grid de todos los gráficos (por explainer)
    print("\nGenerando grid de violin plots (métricas)...")
    try:
        fig = plt.figure(figsize=config.FIGSIZE_MULTI)
        plot_all_metrics_grid_violin(all_raw, contrastive_raw, available_metrics, config, 
                             os.path.join(config.OUTPUT_DIR, "03_violin_grid.png"))
        plt.close(fig)
    except Exception as e:
        print(f"✗ Error generando grid: {e}")
    
    # Gráficos comparando base vs robust por dataset (4 violines por dataset)
    print("\nGenerando violin plots Base vs Robust por dataset...")
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
            plot_metric_base_vs_robust_by_dataset(all_raw, contrastive_raw, base_metric, robust_metric, config, ax)
            metric_name = config.METRIC_NAMES.get(base_metric, base_metric.replace('base_counterfactual_', '')).replace(' ', '_')
            save_path = os.path.join(config.OUTPUT_DIR, f"04_violin_base_vs_robust_{metric_name}.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ {save_path}")
        except Exception as e:
            print(f"✗ Error generando gráfico base vs robust para {base_metric}: {e}")
    
    # Grid de base vs robust
    print("\nGenerando grid de violin plots Base vs Robust...")
    try:
        plot_all_metrics_base_vs_robust_grid(all_raw, contrastive_raw, config,
                                              os.path.join(config.OUTPUT_DIR, "05_violin_base_vs_robust_grid.png"))
    except Exception as e:
        print(f"✗ Error generando grid base vs robust: {e}")
    
    # 4. Exportar resultados
    print("\n[PASO 4] Exportando resultados agregados...")
    print("-" * 80)
    export_aggregated_results(all_raw, contrastive_raw, available_metrics, config)
    
    # 5. Tests estadísticos
    print("\n[PASO 5] Realizando tests estadísticos...")
    print("-" * 80)
    export_statistical_tests(all_raw, contrastive_raw, available_metrics, config)
    
    print("\n" + "=" * 80)
    print(f"✓ COMPLETADO. Resultados en: {config.OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
