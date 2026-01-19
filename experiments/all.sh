#!/bin/bash
#SBATCH --job-name=betarce_all
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=all.log
#SBATCH --error=all.log
#SBATCH --mem=16G

# ==============================================================================
# Experiment Script - All classifiers
# ==============================================================================

# Hydra launcher: joblib (local) or submitit_slurm (cluster)
HYDRA_LAUNCHER=${HYDRA_LAUNCHER:-joblib}

# Get the parent directory (root of the repo)
REPO_ROOT="./"
echo "Repo root: $REPO_ROOT"

EXPERIMENT_NAME="all"
CONFIG_FILENAME="config_exp"

# PATHS - relative to repo root
BASE_PATH="results/$EXPERIMENT_NAME/"
MODEL_PATH="results/$EXPERIMENT_NAME/models/"
LOG_PATH="results/$EXPERIMENT_NAME/logs/"
RESULT_PATH="results/$EXPERIMENT_NAME/results/"

# Create directories if they don't exist
mkdir -p "$REPO_ROOT/$MODEL_PATH" "$REPO_ROOT/$LOG_PATH" "$REPO_ROOT/$RESULT_PATH"

# SWEEP
robust_method=[robx],[betarob]
base_cf=[gs],[dice],[face]
e2e_explainers=[]
datasets=[car_eval],[rice],[wine_quality],[fico],[diabetes],[breast_cancer]
ex_type=[Architecture],[Bootstrap],[Seed]
model_type_to_use=[neural_network],[logistic_regression],[lightgbm]

echo "Running experiment: $EXPERIMENT_NAME"

# Setup conda environment
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"


conda activate "$REPO_ROOT/.conda/"

echo "Starting now ..."

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
