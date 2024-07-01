#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:prjs2e21
#SBATCH --job-name=reg
#SBATCH --output=/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/s3df_runtime_outputs/output-%j.txt
#SBATCH --error=/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/s3df_runtime_outputs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32g
#SBATCH --time=0-24:00:00
#SBATCH --gpus 4
# source ~/.bashrc
# source /sdf/group/lcls/ds/tools/conda_envs/jackh_pytorch/bin/activate cookie_ml

echo starting run 1 at: `date`
# Check which GPU is being used
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# Run the Python script with the specified arguments
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
# python3 /sdf/home/j/jhirschm/COOKIE_ML/src/denoising/ximg_to_ypdf_autoencoder_straight_training.py 
# python3 /sdf/home/j/jhirschm/COOKIE_ML/src/denoising/ximg_to_ypdf_autoencoder_straight_training_fineTuning.py 

python3 /sdf/home/j/jhirschm/COOKIE_ML/src/denoising/ximg_to_ypdf_autoencoder_straight_evaluation.py 

echo Finished at: `date`