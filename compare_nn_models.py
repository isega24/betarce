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
        "nn_standard": "trial/results",           # neural_network
        "nn_contrastive": "contrastive_trial/results"  # neural_network_contrastive
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


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN (Violin Plots con Seaborn)
# ============================================================================

def plot_metric_violin_by_explainer(
    nn_standard: pd.DataFrame,
    nn_contrastive: pd.DataFrame,
    metric: str,
    config: Config,
    ax: Optional[plt.Axes] = None
) -> Optional[plt.Axes]:
    """
    Crea violin plot comparativo de una métrica: NN Standard vs NN Contrastive.
    """
    
    # Preparar datos
    nn_standard_copy = nn_standard.copy()
    nn_standard_copy['Modelo'] = config.EXPERIMENT_LABELS['nn_standard']
    
    nn_contrastive_copy = nn_contrastive.copy()
    nn_contrastive_copy['Modelo'] = config.EXPERIMENT_LABELS['nn_contrastive']
    
    # Combinar datos
    combined = pd.concat([nn_standard_copy, nn_contrastive_copy], ignore_index=True)
    
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
        hue='Modelo',
        ax=ax,
        palette="Set2",
        inner="box"
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
    ax: Optional[plt.Axes] = None
) -> Optional[plt.Axes]:
    """
    Crea violin plot comparativo de una métrica por dataset.
    """
    
    # Preparar datos
    nn_standard_copy = nn_standard.copy()
    nn_standard_copy['Modelo'] = config.EXPERIMENT_LABELS['nn_standard']
    
    nn_contrastive_copy = nn_contrastive.copy()
    nn_contrastive_copy['Modelo'] = config.EXPERIMENT_LABELS['nn_contrastive']
    
    # Combinar datos
    combined = pd.concat([nn_standard_copy, nn_contrastive_copy], ignore_index=True)
    
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
        hue='Modelo',
        ax=ax,
        palette="Set2",
        inner="box"
    )
    
    # Configurar ejes
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} - NN Standard vs NN Contrastive por Dataset', fontsize=13, fontweight='bold')
    ax.legend(title='Modelo', title_fontsize=11, fontsize=10, loc='best', bbox_to_anchor=(1.05, 1))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return ax


def plot_all_metrics_grid_violin(
    nn_standard: pd.DataFrame,
    nn_contrastive: pd.DataFrame,
    metrics: List[str],
    config: Config,
    save_path: Optional[str] = None
) -> None:
    """
    Crea una matriz de violin plots con todas las métricas.
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
            hue='Modelo',
            ax=ax,
            palette="Set2",
            inner="box"
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
    
    plt.show()


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
    
    # Definir métricas a usar
    all_metrics = config.BASE_METRICS + config.ROBUST_METRICS + ['accuracy']
    
    # Filtrar solo métricas disponibles
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
    print("\n[PASO 3] Generando visualizaciones con Violin Plots...")
    print("-" * 80)
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Gráficos individuales por métrica (agrupados por explainer base)
    print("Generando violin plots por explainer...")
    for metric in available_metrics:
        try:
            fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
            plot_metric_violin_by_explainer(nn_standard_raw, nn_contrastive_raw, metric, config, ax)
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
            plot_metric_violin_by_dataset(nn_standard_raw, nn_contrastive_raw, metric, config, ax)
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
        plot_all_metrics_grid_violin(nn_standard_raw, nn_contrastive_raw, available_metrics, config,
                                     os.path.join(config.OUTPUT_DIR, "03_violin_grid.png"))
        plt.close(fig)
    except Exception as e:
        print(f"✗ Error generando grid: {e}")
    
    # 4. Exportar resultados
    print("\n[PASO 4] Exportando resultados agregados...")
    print("-" * 80)
    export_aggregated_results(nn_standard_raw, nn_contrastive_raw, available_metrics, config)
    
    print("\n" + "=" * 80)
    print(f"✓ COMPLETADO. Resultados en: {config.OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
