#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --gpus 1
#SBATCH --output=/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/s3df_runtime_outputs/output-%j.txt
#SBATCH --error=/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/s3df_runtime_outputs/output-%j.txt
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

#module load cuda

which nvcc
nvcc --version

echo Finished
