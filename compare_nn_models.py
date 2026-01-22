"""
Script para comparar Neural Network vs Neural Network Contrastive
Compara resultados de:
- trial: neural_network
- contrastive_trial: neural_network_contrastive

También incluye accuracy basado en model1_pred_crisp vs y_test_sample
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
        "nn_standard": "all/results",           # neural_network (experimento completo)
        "nn_contrastive": "contrastive/results"  # neural_network_contrastive (experimento completo)
    }
    
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
    
    # Dirección de las métricas: True = mayor es mejor (↑), False = menor es mejor (↓)
    METRIC_DIRECTION = {
        "base_counterfactual_validity": True,           # Mayor es mejor
        "base_counterfactual_proximityL1": False,       # Menor es mejor
        "base_counterfactual_proximityL2": False,       # Menor es mejor
        "base_counterfactual_plausibility": False,      # Menor es mejor (distancia a vecinos)
        "base_counterfactual_discriminative_power": True,  # Mayor es mejor
        "robust_counterfactual_validity": True,
        "robust_counterfactual_proximityL1": False,
        "robust_counterfactual_proximityL2": False,
        "robust_counterfactual_plausibility": False,
        "robust_counterfactual_discriminative_power": True,
        "accuracy": True,                               # Mayor es mejor
        # Nombres cortos también
        "validity": True,
        "proximityL1": False,
        "proximityL2": False,
        "plausibility": False,
        "dpow": True,
        "validity (robust)": True,
        "proximityL1 (robust)": False,
        "proximityL2 (robust)": False,
        "plausibility (robust)": False,
        "dpow (robust)": True,
    }
    
    # Etiquetas para experimentos
    EXPERIMENT_LABELS = {
        "nn_standard": "Neural Network (standard)",
        "nn_contrastive": "Neural Network (contrastive)"
    }
    
    # Colores para experimentos
    COLORS = {
        "nn_standard": "#1f77b4",
        "nn_contrastive": "#ff7f0e"
    }
    
    # Directorio de salida
    OUTPUT_DIR = "results_comparison_nn"
    
    # Figsize para gráficos
    FIGSIZE_SINGLE = (12, 6)
    FIGSIZE_MULTI = (16, 10)


def get_metric_name_with_arrow(metric: str, config: Config) -> str:
    """
    Obtiene el nombre legible de una métrica con flecha indicando dirección.
    
    ↑ = Mayor es mejor
    ↓ = Menor es mejor
    
    Args:
        metric: Nombre de la métrica (puede ser nombre completo o corto)
        config: Objeto de configuración
        
    Returns:
        Nombre de la métrica con flecha de dirección
    """
    # Obtener nombre legible
    metric_name = config.METRIC_NAMES.get(metric, metric.replace('base_counterfactual_', '').replace('robust_counterfactual_', ''))
    
    # Determinar dirección (buscar tanto en nombre completo como en nombre corto)
    is_higher_better = config.METRIC_DIRECTION.get(metric, config.METRIC_DIRECTION.get(metric_name, None))
    
    if is_higher_better is True:
        arrow = "↑"
    elif is_higher_better is False:
        arrow = "↓"
    else:
        arrow = ""
    
    return f"{metric_name} {arrow}".strip()


# ============================================================================
# FUNCIONES DE CARGA Y PROCESAMIENTO
# ============================================================================

def find_result_files(exp_type: str, config: Config) -> List[str]:
    """Encuentra todos los archivos de resultados para un tipo de experimento."""
    base_path = os.path.join(config.RESULTS_BASE, config.EXPERIMENT_DIRS[exp_type])
    result_files = glob.glob(
        os.path.join(base_path, "**", "results", "*.csv"),
        recursive=True
    )
    print(f"[{exp_type.upper()}] Encontrados {len(result_files)} archivos de resultados")
    return result_files


def load_and_filter_nn_results(result_files: List[str], exp_type: str, config: Config) -> pd.DataFrame:
    """
    Carga resultados y filtra solo neural_network o neural_network_contrastive.
    """
    dfs = []
    
    # Determinar qué clasificador esperar según el tipo de experimento
    if exp_type == "nn_standard":
        target_classifier = "neural_network"
    else:
        target_classifier = "neural_network_contrastive"
    
    for file_path in result_files:
        try:
            df = pd.read_csv(file_path)
            # Filtrar solo el clasificador específico
            df_filtered = df[df['model_type_to_use'] == target_classifier].copy()
            if len(df_filtered) > 0:
                dfs.append(df_filtered)
        except Exception as e:
            print(f"Advertencia: No se pudo cargar {file_path}: {e}")
            continue
    
    if not dfs:
        print(f"Advertencia: No hay datos para {exp_type}")
        return pd.DataFrame()
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['experiment_type_nn'] = exp_type
    
    print(f"[{exp_type.upper()}] Datos cargados: {len(combined_df)} filas")
    
    return combined_df


def calculate_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula accuracy comparando model1_pred_crisp con y_test_sample.
    """
    df_copy = df.copy()
    df_copy['accuracy'] = (df_copy['model1_pred_crisp'] == df_copy['y_test_sample']).astype(int)
    return df_copy


def prepare_metrics_for_analysis(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Prepara los datos para análisis renombrando columnas de métricas."""
    df_copy = df.copy()
    
    if 'base_counterfactual_discriminative_power' in df_copy.columns:
        df_copy['base_counterfactual_dpow'] = df_copy['base_counterfactual_discriminative_power']
    
    if 'robust_counterfactual_discriminative_power' in df_copy.columns:
        df_copy['robust_counterfactual_dpow'] = df_copy['robust_counterfactual_discriminative_power']
    
    return df_copy


def remove_outliers_iqr(df: pd.DataFrame, column: str, k: float = 3.0) -> pd.DataFrame:
    """
    Elimina outliers de un DataFrame basándose en el rango intercuartílico (IQR).
    
    Outliers se definen como valores fuera del rango [Q1 - k*IQR, Q3 + k*IQR].
    
    Args:
        df: DataFrame de entrada
        column: Nombre de la columna a filtrar
        k: Factor multiplicador del IQR (por defecto 3.0 para outliers extremos)
        
    Returns:
        DataFrame filtrado sin outliers
    """
    if column not in df.columns:
        return df
    
    data = df[column].dropna()
    if len(data) == 0:
        return df
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    
    # Filtrar manteniendo NaN (no son outliers, simplemente no tienen dato)
    mask = (df[column].isna()) | ((df[column] >= lower_bound) & (df[column] <= upper_bound))
    
    n_removed = len(df) - mask.sum()
    if n_removed > 0:
        print(f"  [IQR Filter] {column}: eliminados {n_removed} outliers (bounds: [{lower_bound:.4f}, {upper_bound:.4f}])")
    
    return df[mask].copy()


def remove_outliers_from_combined(df: pd.DataFrame, value_column: str, k: float = 3.0) -> pd.DataFrame:
    """
    Elimina outliers de un DataFrame combinado (formato largo) basándose en IQR.
    
    Args:
        df: DataFrame combinado con columna de valores
        value_column: Nombre de la columna con los valores a filtrar
        k: Factor multiplicador del IQR (por defecto 3.0)
        
    Returns:
        DataFrame filtrado sin outliers
    """
    if value_column not in df.columns:
        return df
    
    data = df[value_column].dropna()
    if len(data) == 0:
        return df
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    
    # Filtrar manteniendo NaN
    mask = (df[value_column].isna()) | ((df[value_column] >= lower_bound) & (df[value_column] <= upper_bound))
    
    n_removed = len(df) - mask.sum()
    if n_removed > 0:
        print(f"  [IQR Filter] {value_column}: eliminados {n_removed} outliers (bounds: [{lower_bound:.4f}, {upper_bound:.4f}])")
    
    return df[mask].copy()


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN (Violin Plots con Seaborn)
# ============================================================================

def plot_metric_violin_by_explainer(
    nn_standard: pd.DataFrame,
    nn_contrastive: pd.DataFrame,
    metric: str,
    config: Config,
    ax: Optional[plt.Axes] = None,
    remove_outliers: bool = True,
    iqr_k: float = 3.0
) -> Optional[plt.Axes]:
    """
    Crea violin plot comparativo de una métrica: NN Standard vs NN Contrastive.
    
    Args:
        remove_outliers: Si True, elimina outliers fuera de [Q1-k*IQR, Q3+k*IQR]
        iqr_k: Factor multiplicador del IQR para definir outliers (default 3.0)
    """
    
    # Preparar datos
    nn_standard_copy = nn_standard.copy()
    nn_standard_copy['Modelo'] = config.EXPERIMENT_LABELS['nn_standard']
    
    nn_contrastive_copy = nn_contrastive.copy()
    nn_contrastive_copy['Modelo'] = config.EXPERIMENT_LABELS['nn_contrastive']
    
    # Combinar datos
    combined = pd.concat([nn_standard_copy, nn_contrastive_copy], ignore_index=True)
    
    # Eliminar outliers si está habilitado
    if remove_outliers and metric in combined.columns:
        combined = remove_outliers_iqr(combined, metric, k=iqr_k)
    
    # Crear gráfico si no existe
    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
    
    # Nombre limpio de la métrica con flecha de dirección
    metric_name = get_metric_name_with_arrow(metric, config)
    
    # Violin plot
    sns.violinplot(
        data=combined,
        x='base_cf_method',
        y=metric,
        hue='Modelo',
        ax=ax,
        palette="Set2",
        inner="box",
        cut=0
    )
    
    # Configurar ejes
    ax.set_xlabel('Explainer Base', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} - NN Standard vs NN Contrastive', fontsize=13, fontweight='bold')
    ax.legend(title='Modelo', title_fontsize=11, fontsize=10, loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return ax


def plot_metric_violin_by_dataset(
    nn_standard: pd.DataFrame,
    nn_contrastive: pd.DataFrame,
    metric: str,
    config: Config,
    ax: Optional[plt.Axes] = None,
    remove_outliers: bool = True,
    iqr_k: float = 3.0
) -> Optional[plt.Axes]:
    """
    Crea violin plot comparativo de una métrica por dataset.
    
    Args:
        remove_outliers: Si True, elimina outliers fuera de [Q1-k*IQR, Q3+k*IQR]
        iqr_k: Factor multiplicador del IQR para definir outliers (default 3.0)
    """
    
    # Preparar datos
    nn_standard_copy = nn_standard.copy()
    nn_standard_copy['Modelo'] = config.EXPERIMENT_LABELS['nn_standard']
    
    nn_contrastive_copy = nn_contrastive.copy()
    nn_contrastive_copy['Modelo'] = config.EXPERIMENT_LABELS['nn_contrastive']
    
    # Combinar datos
    combined = pd.concat([nn_standard_copy, nn_contrastive_copy], ignore_index=True)
    
    # Eliminar outliers si está habilitado
    if remove_outliers and metric in combined.columns:
        combined = remove_outliers_iqr(combined, metric, k=iqr_k)
    
    # Crear gráfico si no existe
    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
    
    # Nombre limpio de la métrica con flecha de dirección
    metric_name = get_metric_name_with_arrow(metric, config)
    
    # Violin plot
    sns.violinplot(
        data=combined,
        x='dataset_name',
        y=metric,
        hue='Modelo',
        ax=ax,
        palette="Set2",
        inner="box",
        cut=0
    )
    
    # Configurar ejes
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} - NN Standard vs NN Contrastive por Dataset', fontsize=13, fontweight='bold')
    ax.legend(title='Modelo', title_fontsize=11, fontsize=10, loc='best', bbox_to_anchor=(1.05, 1))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return ax


def plot_metric_base_vs_robust_by_dataset(
    nn_standard: pd.DataFrame,
    nn_contrastive: pd.DataFrame,
    base_metric: str,
    robust_metric: str,
    config: Config,
    ax: Optional[plt.Axes] = None,
    remove_outliers: bool = True,
    iqr_k: float = 3.0
) -> Optional[plt.Axes]:
    """
    Crea violin plot comparativo de una métrica base vs su versión robust,
    para ambos modelos (NN Standard y NN Contrastive) agrupado por dataset.
    
    Muestra 4 violines por dataset:
    - NN Standard (base)
    - NN Standard (robust)
    - NN Contrastive (base)
    - NN Contrastive (robust)
    
    Args:
        remove_outliers: Si True, elimina outliers fuera de [Q1-k*IQR, Q3+k*IQR]
        iqr_k: Factor multiplicador del IQR para definir outliers (default 3.0)
    """
    
    # Verificar que las métricas existen
    if base_metric not in nn_standard.columns or robust_metric not in nn_standard.columns:
        print(f"Advertencia: Métricas {base_metric} o {robust_metric} no encontradas en nn_standard")
        return None
    if base_metric not in nn_contrastive.columns or robust_metric not in nn_contrastive.columns:
        print(f"Advertencia: Métricas {base_metric} o {robust_metric} no encontradas en nn_contrastive")
        return None
    
    # Preparar datos - transformar a formato largo para comparar base vs robust
    rows = []
    
    # NN Standard - base
    for _, row in nn_standard.iterrows():
        rows.append({
            'dataset_name': row['dataset_name'],
            'value': row[base_metric],
            'Variante': 'NN Standard (base)'
        })
    
    # NN Standard - robust
    for _, row in nn_standard.iterrows():
        rows.append({
            'dataset_name': row['dataset_name'],
            'value': row[robust_metric],
            'Variante': 'NN Standard (robust)'
        })
    
    # NN Contrastive - base
    for _, row in nn_contrastive.iterrows():
        rows.append({
            'dataset_name': row['dataset_name'],
            'value': row[base_metric],
            'Variante': 'NN Contrastive (base)'
        })
    
    # NN Contrastive - robust
    for _, row in nn_contrastive.iterrows():
        rows.append({
            'dataset_name': row['dataset_name'],
            'value': row[robust_metric],
            'Variante': 'NN Contrastive (robust)'
        })
    
    combined = pd.DataFrame(rows)
    
    # Eliminar outliers si está habilitado
    if remove_outliers:
        combined = remove_outliers_from_combined(combined, 'value', k=iqr_k)
    
    # Crear gráfico si no existe
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))
    
    # Nombre limpio de la métrica con flecha de dirección
    metric_name = get_metric_name_with_arrow(base_metric, config)
    
    # Paleta de colores: azul para standard, naranja para contrastive, más claro para base, más oscuro para robust
    palette = {
        'NN Standard (base)': '#7fbfff',      # azul claro
        'NN Standard (robust)': '#1f77b4',    # azul oscuro
        'NN Contrastive (base)': '#ffbf7f',   # naranja claro
        'NN Contrastive (robust)': '#ff7f0e'  # naranja oscuro
    }
    
    # Orden para las variantes
    hue_order = ['NN Standard (base)', 'NN Standard (robust)', 'NN Contrastive (base)', 'NN Contrastive (robust)']
    
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
    ax.set_title(f'{metric_name} - Base vs Robust por Modelo y Dataset', fontsize=13, fontweight='bold')
    ax.legend(title='Variante', title_fontsize=10, fontsize=9, loc='best', bbox_to_anchor=(1.05, 1))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    return ax


def plot_all_metrics_base_vs_robust_grid(
    nn_standard: pd.DataFrame,
    nn_contrastive: pd.DataFrame,
    config: Config,
    save_path: Optional[str] = None,
    remove_outliers: bool = True,
    iqr_k: float = 3.0
) -> None:
    """
    Crea una matriz de violin plots comparando base vs robust para todas las métricas,
    agrupado por dataset.
    
    Args:
        remove_outliers: Si True, elimina outliers fuera de [Q1-k*IQR, Q3+k*IQR]
        iqr_k: Factor multiplicador del IQR para definir outliers (default 3.0)
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
        if b in nn_standard.columns and r in nn_standard.columns 
        and b in nn_contrastive.columns and r in nn_contrastive.columns
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
        'NN Standard (base)': '#7fbfff',
        'NN Standard (robust)': '#1f77b4',
        'NN Contrastive (base)': '#ffbf7f',
        'NN Contrastive (robust)': '#ff7f0e'
    }
    hue_order = ['NN Standard (base)', 'NN Standard (robust)', 'NN Contrastive (base)', 'NN Contrastive (robust)']
    
    for idx, (base_metric, robust_metric) in enumerate(available_pairs):
        ax = axes[idx]
        
        # Preparar datos
        rows = []
        for _, row in nn_standard.iterrows():
            rows.append({'dataset_name': row['dataset_name'], 'value': row[base_metric], 'Variante': 'NN Standard (base)'})
            rows.append({'dataset_name': row['dataset_name'], 'value': row[robust_metric], 'Variante': 'NN Standard (robust)'})
        for _, row in nn_contrastive.iterrows():
            rows.append({'dataset_name': row['dataset_name'], 'value': row[base_metric], 'Variante': 'NN Contrastive (base)'})
            rows.append({'dataset_name': row['dataset_name'], 'value': row[robust_metric], 'Variante': 'NN Contrastive (robust)'})
        
        combined = pd.DataFrame(rows)
        
        # Eliminar outliers si está habilitado
        if remove_outliers:
            combined = remove_outliers_from_combined(combined, 'value', k=iqr_k)
        
        # Nombre limpio de la métrica con flecha de dirección
        metric_name = get_metric_name_with_arrow(base_metric, config)
        
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
    
    fig.suptitle('Comparación Base vs Robust: NN Standard vs NN Contrastive por Dataset', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado: {save_path}")
    
    plt.close(fig)


def plot_accuracy_by_dataset(
    nn_standard: pd.DataFrame,
    nn_contrastive: pd.DataFrame,
    config: Config,
    ax: Optional[plt.Axes] = None
) -> Optional[plt.Axes]:
    """
    Crea gráfico de barras de accuracy agrupado por dataset.
    """
    
    # Calcular accuracy promedio por dataset para cada modelo
    accuracy_standard = nn_standard.groupby('dataset_name', observed=True)['accuracy'].mean()
    accuracy_contrastive = nn_contrastive.groupby('dataset_name', observed=True)['accuracy'].mean()
    
    # Obtener datasets comunes
    common_datasets = sorted(set(accuracy_standard.index) & set(accuracy_contrastive.index))
    
    if not common_datasets:
        print("Advertencia: Sin datasets comunes para accuracy")
        return None
    
    # Crear gráfico si no existe
    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
    
    # Preparar datos
    x = np.arange(len(common_datasets))
    width = 0.35
    
    accuracies_standard = [accuracy_standard[d] for d in common_datasets]
    accuracies_contrastive = [accuracy_contrastive[d] for d in common_datasets]
    
    # Barras
    bars1 = ax.bar(x - width/2, accuracies_standard, width, 
                   label=config.EXPERIMENT_LABELS['nn_standard'],
                   color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, accuracies_contrastive, width,
                   label=config.EXPERIMENT_LABELS['nn_contrastive'],
                   color='#ff7f0e', alpha=0.8)
    
    # Agregar valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Configurar ejes
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Promedio por Dataset', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(common_datasets, rotation=45, ha='right')
    ax.legend(fontsize=11, loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    return ax


def plot_all_metrics_grid_violin(
    nn_standard: pd.DataFrame,
    nn_contrastive: pd.DataFrame,
    metrics: List[str],
    config: Config,
    save_path: Optional[str] = None,
    remove_outliers: bool = True,
    iqr_k: float = 3.0
) -> None:
    """
    Crea una matriz de violin plots con todas las métricas.
    
    Args:
        remove_outliers: Si True, elimina outliers fuera de [Q1-k*IQR, Q3+k*IQR]
        iqr_k: Factor multiplicador del IQR para definir outliers (default 3.0)
    """
    
    # Filtrar métricas que existen
    available_metrics = [m for m in metrics if m in nn_standard.columns and m in nn_contrastive.columns]
    
    if not available_metrics:
        print("Advertencia: No hay métricas disponibles")
        return
    
    # Preparar datos
    nn_standard_copy = nn_standard.copy()
    nn_standard_copy['Modelo'] = config.EXPERIMENT_LABELS['nn_standard']
    
    nn_contrastive_copy = nn_contrastive.copy()
    nn_contrastive_copy['Modelo'] = config.EXPERIMENT_LABELS['nn_contrastive']
    
    combined = pd.concat([nn_standard_copy, nn_contrastive_copy], ignore_index=True)
    
    # Eliminar outliers para cada métrica si está habilitado
    if remove_outliers:
        for metric in available_metrics:
            if metric in combined.columns:
                combined = remove_outliers_iqr(combined, metric, k=iqr_k)
    
    # Crear grid
    n_cols = 3
    n_rows = (len(available_metrics) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(config.FIGSIZE_MULTI[0], n_rows * 5))
    axes = axes.flatten()
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        # Nombre limpio de la métrica con flecha de dirección
        metric_name = get_metric_name_with_arrow(metric, config)
        
        sns.violinplot(
            data=combined,
            x='base_cf_method',
            y=metric,
            hue='Modelo',
            ax=ax,
            palette="Set2",
            inner="box",
            cut=0
        )
        
        ax.set_xlabel('Explainer Base', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
        ax.legend(title='Modelo', fontsize=9, title_fontsize=10, loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Ocultar ejes no usados
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Comparación: Neural Network Standard vs Neural Network Contrastive', 
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
    nn_standard: pd.DataFrame,
    nn_contrastive: pd.DataFrame,
    metrics: List[str],
    config: Config
) -> pd.DataFrame:
    """
    Realiza tests estadísticos para comparar métricas entre grupos.
    
    Tests utilizados:
    - Mann-Whitney U: Para comparar grupos independientes (NN Standard vs NN Contrastive)
    - Wilcoxon signed-rank: Para comparar métricas base vs robust (datos pareados)
    
    Args:
        nn_standard: DataFrame con resultados de NN Standard
        nn_contrastive: DataFrame con resultados de NN Contrastive
        metrics: Lista de métricas a comparar
        config: Objeto de configuración
        
    Returns:
        DataFrame con resultados de los tests estadísticos
    """
    results = []
    
    available_metrics = [m for m in metrics if m in nn_standard.columns and m in nn_contrastive.columns]
    
    for metric in available_metrics:
        # Nombre limpio de la métrica con flecha de dirección
        metric_name = get_metric_name_with_arrow(metric, config)
        
        # Obtener datos
        data_standard = nn_standard[metric].dropna()
        data_contrastive = nn_contrastive[metric].dropna()
        
        if len(data_standard) < 2 or len(data_contrastive) < 2:
            continue
        
        # Mann-Whitney U test (grupos independientes)
        try:
            mw_stat, mw_pvalue = stats.mannwhitneyu(
                data_standard, data_contrastive, alternative='two-sided'
            )
        except Exception:
            mw_stat, mw_pvalue = np.nan, np.nan
        
        # Calcular tamaño del efecto (rank-biserial correlation)
        n1, n2 = len(data_standard), len(data_contrastive)
        if n1 > 0 and n2 > 0:
            effect_size = 1 - (2 * mw_stat) / (n1 * n2)  # rank-biserial correlation
        else:
            effect_size = np.nan
        
        # Estadísticas descriptivas
        mean_standard = data_standard.mean()
        mean_contrastive = data_contrastive.mean()
        std_standard = data_standard.std()
        std_contrastive = data_contrastive.std()
        
        # Obtener si mayor es mejor para esta métrica
        is_higher_better = config.METRIC_DIRECTION.get(
            metric, 
            config.METRIC_DIRECTION.get(metric.replace('base_counterfactual_', '').replace('robust_counterfactual_', ''), True)
        )
        
        # Determinar dirección del efecto considerando qué es "mejor"
        if mw_pvalue < 0.05 if not np.isnan(mw_pvalue) else False:
            if is_higher_better:
                # Mayor es mejor
                if mean_contrastive > mean_standard:
                    direction = 'NN Contrastive > NN Standard'
                    winner = 'NN Contrastive'
                else:
                    direction = 'NN Standard > NN Contrastive'
                    winner = 'NN Standard'
            else:
                # Menor es mejor
                if mean_contrastive < mean_standard:
                    direction = 'NN Contrastive < NN Standard'
                    winner = 'NN Contrastive'
                else:
                    direction = 'NN Standard < NN Contrastive'
                    winner = 'NN Standard'
        else:
            direction = 'No significant difference'
            winner = 'None'
        
        results.append({
            'metric': metric_name,
            'test': 'Mann-Whitney U',
            'comparison': 'NN Standard vs NN Contrastive',
            'n_standard': len(data_standard),
            'n_contrastive': len(data_contrastive),
            'mean_standard': round(mean_standard, 4),
            'mean_contrastive': round(mean_contrastive, 4),
            'std_standard': round(std_standard, 4),
            'std_contrastive': round(std_contrastive, 4),
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
    
    Usa Wilcoxon signed-rank test (datos pareados, no asume normalidad).
    
    Args:
        df: DataFrame con ambas métricas (base y robust)
        base_metrics: Lista de métricas base
        robust_metrics: Lista de métricas robustas correspondientes
        config: Objeto de configuración
        group_name: Nombre del grupo para identificación
        
    Returns:
        DataFrame con resultados de los tests pareados
    """
    results = []
    
    for base_metric, robust_metric in zip(base_metrics, robust_metrics):
        if base_metric not in df.columns or robust_metric not in df.columns:
            continue
        
        # Obtener datos pareados (mismas filas)
        mask = df[base_metric].notna() & df[robust_metric].notna()
        data_base = df.loc[mask, base_metric]
        data_robust = df.loc[mask, robust_metric]
        
        if len(data_base) < 2:
            continue
        
        # Nombre limpio de la métrica con flecha de dirección
        metric_name = get_metric_name_with_arrow(base_metric, config)
        
        # Wilcoxon signed-rank test (datos pareados)
        try:
            wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(
                data_base, data_robust, alternative='two-sided'
            )
        except Exception:
            wilcoxon_stat, wilcoxon_pvalue = np.nan, np.nan
        
        # Calcular tamaño del efecto (matched-pairs rank-biserial correlation)
        diff = data_robust - data_base
        n = len(diff[diff != 0])
        if n > 0 and not np.isnan(wilcoxon_stat):
            effect_size = 1 - (2 * wilcoxon_stat) / (n * (n + 1) / 2)
        else:
            effect_size = np.nan
        
        # Estadísticas descriptivas
        mean_base = data_base.mean()
        mean_robust = data_robust.mean()
        mean_diff = (data_robust - data_base).mean()
        
        # Obtener si mayor es mejor para esta métrica
        is_higher_better = config.METRIC_DIRECTION.get(
            base_metric, 
            config.METRIC_DIRECTION.get(base_metric.replace('base_counterfactual_', '').replace('robust_counterfactual_', ''), True)
        )
        
        # Determinar dirección del efecto considerando qué es "mejor"
        if wilcoxon_pvalue < 0.05 if not np.isnan(wilcoxon_pvalue) else False:
            if is_higher_better:
                # Mayor es mejor
                if mean_robust > mean_base:
                    direction = 'Robust > Base'
                    winner = 'Robust'
                else:
                    direction = 'Base > Robust'
                    winner = 'Base'
            else:
                # Menor es mejor
                if mean_robust < mean_base:
                    direction = 'Robust < Base'
                    winner = 'Robust'
                else:
                    direction = 'Base < Robust'
                    winner = 'Base'
        else:
            direction = 'No significant difference'
            winner = 'None'
        
        results.append({
            'metric': metric_name,
            'test': 'Wilcoxon signed-rank',
            'comparison': f'{group_name}: Base vs Robust',
            'n_pairs': len(data_base),
            'mean_base': round(mean_base, 4),
            'mean_robust': round(mean_robust, 4),
            'mean_difference': round(mean_diff, 4),
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
    nn_standard: pd.DataFrame,
    nn_contrastive: pd.DataFrame,
    metrics: List[str],
    config: Config
) -> pd.DataFrame:
    """
    Realiza tests estadísticos por dataset.
    
    Args:
        nn_standard: DataFrame con resultados de NN Standard
        nn_contrastive: DataFrame con resultados de NN Contrastive
        metrics: Lista de métricas
        config: Objeto de configuración
        
    Returns:
        DataFrame con resultados por dataset
    """
    results = []
    
    datasets = set(nn_standard['dataset_name'].unique()) & set(nn_contrastive['dataset_name'].unique())
    available_metrics = [m for m in metrics if m in nn_standard.columns and m in nn_contrastive.columns]
    
    for dataset in sorted(datasets):
        df_std = nn_standard[nn_standard['dataset_name'] == dataset]
        df_con = nn_contrastive[nn_contrastive['dataset_name'] == dataset]
        
        for metric in available_metrics:
            # Nombre limpio de la métrica con flecha de dirección
            metric_name = get_metric_name_with_arrow(metric, config)
            
            data_standard = df_std[metric].dropna()
            data_contrastive = df_con[metric].dropna()
            
            if len(data_standard) < 2 or len(data_contrastive) < 2:
                continue
            
            try:
                mw_stat, mw_pvalue = stats.mannwhitneyu(
                    data_standard, data_contrastive, alternative='two-sided'
                )
            except Exception:
                mw_stat, mw_pvalue = np.nan, np.nan
            
            n1, n2 = len(data_standard), len(data_contrastive)
            effect_size = 1 - (2 * mw_stat) / (n1 * n2) if n1 > 0 and n2 > 0 and not np.isnan(mw_stat) else np.nan
            
            # Obtener si mayor es mejor para esta métrica
            is_higher_better = config.METRIC_DIRECTION.get(
                metric, 
                config.METRIC_DIRECTION.get(metric.replace('base_counterfactual_', '').replace('robust_counterfactual_', ''), True)
            )
            
            # Determinar dirección del efecto considerando qué es "mejor"
            mean_std = data_standard.mean()
            mean_con = data_contrastive.mean()
            if mw_pvalue < 0.05 if not np.isnan(mw_pvalue) else False:
                if is_higher_better:
                    # Mayor es mejor
                    if mean_con > mean_std:
                        direction = 'NN Contrastive > NN Standard'
                        winner = 'NN Contrastive'
                    else:
                        direction = 'NN Standard > NN Contrastive'
                        winner = 'NN Standard'
                else:
                    # Menor es mejor
                    if mean_con < mean_std:
                        direction = 'NN Contrastive < NN Standard'
                        winner = 'NN Contrastive'
                    else:
                        direction = 'NN Standard < NN Contrastive'
                        winner = 'NN Standard'
            else:
                direction = 'No significant difference'
                winner = 'None'
            
            results.append({
                'dataset': dataset,
                'metric': metric_name,
                'n_standard': n1,
                'n_contrastive': n2,
                'mean_standard': round(mean_std, 4),
                'mean_contrastive': round(mean_con, 4),
                'p_value': mw_pvalue,
                'effect_size': round(effect_size, 4) if not np.isnan(effect_size) else np.nan,
                'significant_0.05': mw_pvalue < 0.05 if not np.isnan(mw_pvalue) else False,
                'direction': direction,
                'winner': winner
            })
    
    return pd.DataFrame(results)


def export_statistical_tests(
    nn_standard: pd.DataFrame,
    nn_contrastive: pd.DataFrame,
    metrics: List[str],
    config: Config
) -> None:
    """
    Exporta todos los tests estadísticos a CSV.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Test Mann-Whitney global
    print("\n  Realizando tests Mann-Whitney U (NN Standard vs NN Contrastive)...")
    global_tests = perform_statistical_tests(nn_standard, nn_contrastive, metrics, config)
    if not global_tests.empty:
        global_tests.to_csv(os.path.join(config.OUTPUT_DIR, "statistical_tests_global.csv"), index=False)
        print(f"  ✓ Tests globales guardados")
        
        # Mostrar resumen
        sig_count = global_tests['significant_0.05'].sum()
        total = len(global_tests)
        pct = (sig_count / total * 100) if total > 0 else 0
        print(f"  Métricas con diferencias significativas (p<0.05): {sig_count}/{total} ({pct:.1f}%)")
    
    # Tests por dataset
    print("\n  Realizando tests por dataset...")
    dataset_tests = perform_statistical_tests_by_dataset(nn_standard, nn_contrastive, metrics, config)
    if not dataset_tests.empty:
        dataset_tests.to_csv(os.path.join(config.OUTPUT_DIR, "statistical_tests_by_dataset.csv"), index=False)
        print(f"  ✓ Tests por dataset guardados")
    
    # Tests pareados (base vs robust) para cada grupo
    base_metrics = config.BASE_METRICS
    robust_metrics = config.ROBUST_METRICS
    
    print("\n  Realizando tests Wilcoxon pareados (Base vs Robust)...")
    
    # Para NN Standard
    paired_standard = perform_paired_statistical_tests(
        nn_standard, base_metrics, robust_metrics, config, "NN Standard"
    )
    
    # Para NN Contrastive
    paired_contrastive = perform_paired_statistical_tests(
        nn_contrastive, base_metrics, robust_metrics, config, "NN Contrastive"
    )
    
    paired_all = pd.concat([paired_standard, paired_contrastive], ignore_index=True)
    if not paired_all.empty:
        paired_all.to_csv(os.path.join(config.OUTPUT_DIR, "statistical_tests_paired_base_vs_robust.csv"), index=False)
        print(f"  ✓ Tests pareados guardados")
    
    # Crear resumen visual
    print("\n  === RESUMEN DE TESTS ESTADÍSTICOS ===")
    if not global_tests.empty:
        print("\n  Comparación NN Standard vs NN Contrastive (Mann-Whitney U):")
        for _, row in global_tests.iterrows():
            sig = "***" if row['significant_0.001'] else ("**" if row['significant_0.01'] else ("*" if row['significant_0.05'] else ""))
            direction_str = f" → {row['winner']}" if row['winner'] != 'None' else ""
            effect_size = row.get('effect_size', np.nan)
            print(f"    {row['metric']}: p={row['p_value']:.4e} {sig}{direction_str}, r = {effect_size}")
        
        # Resumen de ganadores
        winners = global_tests[global_tests['winner'] != 'None']['winner'].value_counts()
        print(f"\n  Resumen de diferencias significativas:")
        for winner, count in winners.items():
            print(f"    {winner}: {count} métricas")
    
    if not paired_all.empty:
        print("\n  Comparación Base vs Robust (Wilcoxon signed-rank):")
        for _, row in paired_all.iterrows():
            sig = "***" if row['significant_0.001'] else ("**" if row['significant_0.01'] else ("*" if row['significant_0.05'] else ""))
            direction_str = f" → {row['winner']}" if row['winner'] != 'None' else ""
            effect_size = row.get('effect_size', np.nan)
            print(f"    {row['comparison']} - {row['metric']}: p={row['p_value']:.4e} {sig}{direction_str}, r = {effect_size}")
    
    print("\n  Leyenda: * p<0.05, ** p<0.01, *** p<0.001")
    print("\n            r = effect size: r < 0.3: pequeño, 0.3 <= r < 0.5: mediano, 0.5 <= r < 1: grande")


# ============================================================================
# FUNCIONES DE EXPORTACIÓN
# ============================================================================

def export_aggregated_results(
    nn_standard: pd.DataFrame,
    nn_contrastive: pd.DataFrame,
    metrics: List[str],
    config: Config
) -> None:
    """Exporta resultados agregados a CSV."""
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Filtrar métricas disponibles
    available_metrics = [m for m in metrics if m in nn_standard.columns and m in nn_contrastive.columns]
    
    # Agregación por explainer base
    nn_standard_by_explainer = nn_standard.groupby('base_cf_method', observed=True)[available_metrics].mean().round(4)
    nn_standard_by_explainer['modelo'] = 'nn_standard'
    nn_standard_by_explainer.to_csv(os.path.join(config.OUTPUT_DIR, "aggregated_by_explainer_nn_standard.csv"))
    
    nn_contrastive_by_explainer = nn_contrastive.groupby('base_cf_method', observed=True)[available_metrics].mean().round(4)
    nn_contrastive_by_explainer['modelo'] = 'nn_contrastive'
    nn_contrastive_by_explainer.to_csv(os.path.join(config.OUTPUT_DIR, "aggregated_by_explainer_nn_contrastive.csv"))
    
    # Agregación por dataset
    nn_standard_by_dataset = nn_standard.groupby('dataset_name', observed=True)[available_metrics].mean().round(4)
    nn_standard_by_dataset['modelo'] = 'nn_standard'
    nn_standard_by_dataset.to_csv(os.path.join(config.OUTPUT_DIR, "aggregated_by_dataset_nn_standard.csv"))
    
    nn_contrastive_by_dataset = nn_contrastive.groupby('dataset_name', observed=True)[available_metrics].mean().round(4)
    nn_contrastive_by_dataset['modelo'] = 'nn_contrastive'
    nn_contrastive_by_dataset.to_csv(os.path.join(config.OUTPUT_DIR, "aggregated_by_dataset_nn_contrastive.csv"))
    
    # Accuracy agregado
    if 'accuracy' in nn_standard.columns:
        accuracy_nn_standard = nn_standard.groupby('dataset_name', observed=True)['accuracy'].mean().round(4)
        accuracy_nn_standard.to_csv(os.path.join(config.OUTPUT_DIR, "accuracy_by_dataset_nn_standard.csv"))
        
        accuracy_nn_contrastive = nn_contrastive.groupby('dataset_name', observed=True)['accuracy'].mean().round(4)
        accuracy_nn_contrastive.to_csv(os.path.join(config.OUTPUT_DIR, "accuracy_by_dataset_nn_contrastive.csv"))
    
    print(f"\nResultados agregados exportados a {config.OUTPUT_DIR}/")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal del script"""
    
    print("=" * 80)
    print("COMPARACIÓN: Neural Network Standard vs Neural Network Contrastive")
    print("=" * 80)
    
    config = Config()
    
    # 1. Encontrar y cargar archivos de resultados
    print("\n[PASO 1] Cargando resultados...")
    print("-" * 80)
    
    nn_standard_files = find_result_files("nn_standard", config)
    nn_contrastive_files = find_result_files("nn_contrastive", config)
    
    nn_standard_raw = load_and_filter_nn_results(nn_standard_files, "nn_standard", config)
    nn_contrastive_raw = load_and_filter_nn_results(nn_contrastive_files, "nn_contrastive", config)
    
    if nn_standard_raw.empty or nn_contrastive_raw.empty:
        print("ERROR: No se pudieron cargar los datos. Abortando.")
        return
    
    # 2. Procesar datos
    print("\n[PASO 2] Procesando datos...")
    print("-" * 80)
    
    # Calcular accuracy
    nn_standard_raw = calculate_accuracy(nn_standard_raw)
    nn_contrastive_raw = calculate_accuracy(nn_contrastive_raw)
    
    # Preparar métricas
    nn_standard_raw = prepare_metrics_for_analysis(nn_standard_raw, config)
    nn_contrastive_raw = prepare_metrics_for_analysis(nn_contrastive_raw, config)
    
    # Definir métricas a usar (excluir accuracy de los violin plots)
    all_metrics = config.BASE_METRICS + config.ROBUST_METRICS
    
    # Filtrar solo métricas disponibles (accuracy se visualiza con barras, no violin)
    available_metrics = [m for m in all_metrics if m in nn_standard_raw.columns and m in nn_contrastive_raw.columns]
    print(f"Métricas disponibles para comparar: {len(available_metrics)}")
    for m in available_metrics:
        print(f"  - {config.METRIC_NAMES.get(m, m)}")
    
    print(f"\nDatos NN Standard: {len(nn_standard_raw)} registros")
    print(f"Datos NN Contrastive: {len(nn_contrastive_raw)} registros")
    print(f"Explainers base (NN Standard): {nn_standard_raw['base_cf_method'].unique()}")
    print(f"Explainers base (NN Contrastive): {nn_contrastive_raw['base_cf_method'].unique()}")
    print(f"Datasets (NN Standard): {nn_standard_raw['dataset_name'].unique()}")
    print(f"Datasets (NN Contrastive): {nn_contrastive_raw['dataset_name'].unique()}")
    
    # 3. Crear visualizaciones
    print("\n[PASO 3] Generando visualizaciones...")
    print("-" * 80)
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Gráfico de barras de accuracy por dataset (PRINCIPAL para accuracy)
    print("Generando gráfico de BARRAS de accuracy por dataset...")
    try:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
        plot_accuracy_by_dataset(nn_standard_raw, nn_contrastive_raw, config, ax)
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
            plot_metric_violin_by_explainer(nn_standard_raw, nn_contrastive_raw, metric, config, ax)
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
            plot_metric_violin_by_dataset(nn_standard_raw, nn_contrastive_raw, metric, config, ax)
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
        plot_all_metrics_grid_violin(nn_standard_raw, nn_contrastive_raw, available_metrics, config,
                                     os.path.join(config.OUTPUT_DIR, "04_violin_grid.png"))
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
            plot_metric_base_vs_robust_by_dataset(nn_standard_raw, nn_contrastive_raw, base_metric, robust_metric, config, ax)
            metric_name = config.METRIC_NAMES.get(base_metric, base_metric.replace('base_counterfactual_', '')).replace(' ', '_')
            save_path = os.path.join(config.OUTPUT_DIR, f"05_violin_base_vs_robust_{metric_name}.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ {save_path}")
        except Exception as e:
            print(f"✗ Error generando gráfico base vs robust para {base_metric}: {e}")
    
    # Grid de base vs robust
    print("\nGenerando grid de violin plots Base vs Robust...")
    try:
        plot_all_metrics_base_vs_robust_grid(nn_standard_raw, nn_contrastive_raw, config,
                                              os.path.join(config.OUTPUT_DIR, "06_violin_base_vs_robust_grid.png"))
    except Exception as e:
        print(f"✗ Error generando grid base vs robust: {e}")
    
    # 4. Exportar resultados
    print("\n[PASO 4] Exportando resultados agregados...")
    print("-" * 80)
    export_aggregated_results(nn_standard_raw, nn_contrastive_raw, available_metrics, config)
    
    # 5. Tests estadísticos
    print("\n[PASO 5] Realizando tests estadísticos...")
    print("-" * 80)
    export_statistical_tests(nn_standard_raw, nn_contrastive_raw, available_metrics, config)
    
    print("\n" + "=" * 80)
    print(f"✓ COMPLETADO. Resultados en: {config.OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
