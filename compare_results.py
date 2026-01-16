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
        split=False
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
        inner="box"
    )
    
    # Configurar ejes
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} - Comparación por Dataset', fontsize=13, fontweight='bold')
    ax.legend(title='Experimento', title_fontsize=11, fontsize=10, loc='best', bbox_to_anchor=(1.05, 1))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return ax


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
            inner="box"
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
    
    plt.show()



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
    print("\n[PASO 3] Generando visualizaciones con Violin Plots...")
    print("-" * 80)
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Gráficos individuales por métrica (agrupados por explainer base)
    print("Generando violin plots por explainer...")
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
    print("\nGenerando violin plots por dataset...")
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
    print("\nGenerando grid de violin plots...")
    try:
        fig = plt.figure(figsize=config.FIGSIZE_MULTI)
        plot_all_metrics_grid_violin(all_raw, contrastive_raw, available_metrics, config, 
                             os.path.join(config.OUTPUT_DIR, "03_violin_grid.png"))
        plt.close(fig)
    except Exception as e:
        print(f"✗ Error generando grid: {e}")
    
    # 4. Exportar resultados
    print("\n[PASO 4] Exportando resultados agregados...")
    print("-" * 80)
    export_aggregated_results(all_raw, contrastive_raw, available_metrics, config)
    
    print("\n" + "=" * 80)
    print(f"✓ COMPLETADO. Resultados en: {config.OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
