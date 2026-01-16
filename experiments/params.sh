#!/bin/bash

# SLURM Configuration
# Adjust SLURM_PARTITION to one of: dgx, dios, default
SLURM_PARTITION=${SLURM_PARTITION:-dgx}

# Hydra launcher: joblib (local) or submitit_slurm (cluster)
# Use joblib for local execution, submitit_slurm for SLURM cluster
HYDRA_LAUNCHER=${HYDRA_LAUNCHER:-joblib}

#SBATCH --job-name=betarce_params
#SBATCH --partition=$SLURM_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --output=params.log
#SBATCH --error=params.log
#SBATCH --mem=16G



# Get the parent directory (root of the repo)
REPO_ROOT="./"
echo "Repo root: $REPO_ROOT"

EXPERIMENT_NAME="params"
CONFIG_FILENAME="config_ex_params"

# PATHS - relative to repo root
BASE_PATH="results/$EXPERIMENT_NAME/"
MODEL_PATH="results/$EXPERIMENT_NAME/models/"
LOG_PATH="results/$EXPERIMENT_NAME/logs/"
RESULT_PATH="results/$EXPERIMENT_NAME/results/"

# Create directories if they don't exist
mkdir -p "$REPO_ROOT/$MODEL_PATH" "$REPO_ROOT/$LOG_PATH" "$REPO_ROOT/$RESULT_PATH"

# SWEEP
e2e_explainers=[]

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
    experiments_setup.e2e_explainers=$e2e_explainers \
    general.result_path=$RESULT_PATH \
    general.log_path=$LOG_PATH \
    general.model_path=$MODEL_PATH 
