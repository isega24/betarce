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
        inner="box"
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
        inner="box"
    )
    
    # Configurar ejes
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} - Base vs Robustez por Dataset', fontsize=13, fontweight='bold')
    ax.legend(title='Tipo', title_fontsize=11, fontsize=10, loc='best', bbox_to_anchor=(1.05, 1))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return ax


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
            inner="box"
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
    
    # Definir métricas a usar (base y robustas)
    all_metrics = config.BASE_METRICS + config.ROBUST_METRICS + ['accuracy']
    
    # Filtrar solo métricas disponibles
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
    print("\n[PASO 3] Generando visualizaciones con Violin Plots...")
    print("-" * 80)
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Gráficos individuales por métrica (agrupados por explainer base)
    print("Generando violin plots por explainer...")
    for metric in available_metrics:
        try:
            fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
            plot_metric_violin_by_explainer(base_results, robust_results, metric, config, ax)
            metric_name = config.METRIC_NAMES.get(metric, metric).replace(' ', '_').replace('(', '').replace(')', '')
            save_path = os.path.join(config.OUTPUT_DIR, f"01_violin_explainer_{metric_name}.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ {save_path}")
        except Exception as e:
            print(f"✗ Error generando gráfico para {metric}: {e}")
    
    # Gráficos individuales por métrica (agrupados por dataset)
    print("\nGenerando violin plots por dataset...")
    for metric in available_metrics:
        try:
            fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
            plot_metric_violin_by_dataset(base_results, robust_results, metric, config, ax)
            metric_name = config.METRIC_NAMES.get(metric, metric).replace(' ', '_').replace('(', '').replace(')', '')
            save_path = os.path.join(config.OUTPUT_DIR, f"02_violin_dataset_{metric_name}.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ {save_path}")
        except Exception as e:
            print(f"✗ Error generando gráfico para {metric}: {e}")
    
    # Grid de todos los gráficos
    print("\nGenerando grid de violin plots...")
    try:
        fig = plt.figure(figsize=config.FIGSIZE_MULTI)
        plot_all_metrics_grid_violin(base_results, robust_results, available_metrics, config,
                                     os.path.join(config.OUTPUT_DIR, "03_violin_grid.png"))
        plt.close(fig)
    except Exception as e:
        print(f"✗ Error generando grid: {e}")
    
    # 4. Exportar resultados
    print("\n[PASO 4] Exportando resultados agregados...")
    print("-" * 80)
    export_aggregated_results(base_results, robust_results, available_metrics, config)
    
    print("\n" + "=" * 80)
    print(f"✓ COMPLETADO. Resultados en: {config.OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
