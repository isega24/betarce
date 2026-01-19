#!/bin/bash
#SBATCH --job-name=betarce_trial
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=trial.log
#SBATCH --error=trial.log
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# ==============================================================================
# TRIAL SCRIPT - Fast testing for all classifiers
# ==============================================================================
# Reduced configuration for quick testing:
# - 1 dataset (fico)
# - 2 folds
# - 5 test samples
# - 5 m2 models
# - 1 ex_type (Seed)
# - 1 posthoc explainer (robx)
# - 1 base explainer (gs)
# - 3 classifiers (neural_network, lightgbm, logistic_regression)
#
# Expected iterations: ~150 iterations
# Estimated time: ~10-20 minutes
# ==============================================================================

# Hydra launcher: joblib (local) or submitit_slurm (cluster)
HYDRA_LAUNCHER=${HYDRA_LAUNCHER:-joblib}

# Get the parent directory (root of the repo)
REPO_ROOT="./"
echo "Repo root: $REPO_ROOT"

EXPERIMENT_NAME="trial"
CONFIG_FILENAME="config_exp_trial"

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
model_type_to_use=[neural_network],[logistic_regression],[lightgbm]

echo "=============================================="
echo "TRIAL RUN - Fast testing configuration"
echo "=============================================="
echo "Dataset: fico"
echo "Classifiers: neural_network, lightgbm, logistic_regression"
echo "Ex types: Seed"
echo "Base CF: gs"
echo "Posthoc: robx"
echo "Expected iterations: ~75"
echo "Estimated time: ~5-15 minutes"
echo "=============================================="

# Setup conda environment
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"

conda activate "$REPO_ROOT/.conda/"

echo "Starting TRIAL experiment now ..."

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
echo "TRIAL RUN COMPLETED"
echo "Results saved in: $RESULT_PATH"
echo "=============================================="
