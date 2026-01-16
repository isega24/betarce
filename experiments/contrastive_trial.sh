#!/bin/bash

# ==============================================================================
# TRIAL SCRIPT - Fast testing for Contrastive Learning Neural Network
# ==============================================================================
# Reduced configuration for quick testing:
# - 1 dataset (fico)
# - 1 fold
# - 5 test samples
# - 5 m2 models
# - 1 ex_type (Seed)
# - 1 posthoc explainer (robx)
# - 1 base explainer (gs)
# - Only neural_network_contrastive classifier
#
# Expected iterations: 1 * 1 * 1 * 1 * 1 * 5 * 5 * 1 = 25 iterations
# Estimated time: ~5-10 minutes
# ==============================================================================

# SLURM Configuration
SLURM_PARTITION=${SLURM_PARTITION:-dgx}
HYDRA_LAUNCHER=${HYDRA_LAUNCHER:-joblib}

#SBATCH --job-name=betarce_contrastive_trial
#SBATCH --partition=$SLURM_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --output=contrastive_trial.log
#SBATCH --error=contrastive_trial.log
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# Get the parent directory (root of the repo)
REPO_ROOT="./"
echo "Repo root: $REPO_ROOT"

EXPERIMENT_NAME="contrastive_trial"
CONFIG_FILENAME="config_exp_contrastive_trial"

# PATHS - relative to repo root
BASE_PATH="results/$EXPERIMENT_NAME/"
MODEL_PATH="results/$EXPERIMENT_NAME/models/"
LOG_PATH="results/$EXPERIMENT_NAME/logs/"
RESULT_PATH="results/$EXPERIMENT_NAME/results/"

# Create directories if they don't exist
mkdir -p "$REPO_ROOT/$MODEL_PATH" "$REPO_ROOT/$LOG_PATH" "$REPO_ROOT/$RESULT_PATH"

# SWEEP - Minimal configuration for fast testing
robust_method=[robx]
base_cf=[gs]
e2e_explainers=[]
datasets=[fico]
ex_type=[Seed]
model_type_to_use=[neural_network_contrastive]

echo "=============================================="
echo "CONTRASTIVE TRIAL RUN - Fast testing"
echo "=============================================="
echo "Dataset: fico"
echo "Classifier: neural_network_contrastive"
echo "Ex types: Seed"
echo "Base CF: gs"
echo "Posthoc: robx"
echo "Expected iterations: ~25"
echo "Estimated time: ~5-10 minutes"
echo "=============================================="

# Setup conda environment
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"

conda activate "$REPO_ROOT/.conda/"

echo "Starting CONTRASTIVE TRIAL experiment now ..."

# Change to repo root and run the experiment
cd "$REPO_ROOT"
python experiment_runner.py -cn $CONFIG_FILENAME --multirun \
    hydra/launcher=$HYDRA_LAUNCHER \
    experiments_setup.posthoc_explainers=$robust_method \
    experiments_setup.e2e_explainers=$e2e_explainers \
    experiments_setup.base_explainers=$base_cf \
    experiments_setup.classifiers=$model_type_to_use \
    experiments_setup.ex_types=$ex_type \
    experiments_setup.datasets=$datasets \
    general.result_path=$RESULT_PATH \
    general.log_path=$LOG_PATH \
    general.model_path=$MODEL_PATH 

echo "=============================================="
echo "CONTRASTIVE TRIAL RUN COMPLETED"
echo "Results saved in: $RESULT_PATH"
echo "=============================================="
