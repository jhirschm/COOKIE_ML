#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:prjs2e21
#SBATCH --job-name=reg
#SBATCH --output=output-%j.txt
#SBATCH --error=output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=25g
#SBATCH --time=0-01:0:00
##SBATCH --gpus 3
# source /sdf/group/lcls/ds/tools/conda_envs/jackh_pytorch/bin/activate
# conda activate jh_pytorch

# Define paths to your data files and scalers
FILE_PATHS="/sdf/data/lcls/ds/prj/prjs2e21/results/even-dist_Pulses_03302024/"
SCALER_SAVE_PATH="/sdf/data/lcls/ds/prj/prjs2e21/results/even-dist_Pulses_03302024/Processed_06202024/"
SCALER_NAME="min_max_scaler"
SAVEPATH="/sdf/data/lcls/ds/prj/prjs2e21/results/even-dist_Pulses_03302024/Processed_06202024/"
ENERGY_ELEMENTS=512

# Run the Python script with the specified arguments
python3 /sdf/home/b/bmencer/jack_code/COOKIE_ML/src/ml_backbone/dataloader.py
