#!/bin/bash
#SBATCH --partition=ampere
## SBATCH --account=lcls:prjs2e21
#SBATCH --job-name=reg
#SBATCH --output=/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/s3df_runtime_outputs/output-%j.txt
#SBATCH --error=/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/s3df_runtime_outputs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32g
#SBATCH --time=0-24:00:00
#SBATCH --gpus 1
# source ~/.bashrc
# source /sdf/group/lcls/ds/tools/conda_envs/jackh_pytorch/bin/activate cookie_ml


# Check if the script argument is provided
if [ -z "$1" ]; then
    echo "No script specified. Usage: sbatch s3df_denoising.sh [training|fineTuning|evaluation]"
    exit 1
fi

echo starting run 1 at: `date`
# Check which GPU is being used
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# Run the Python script with the specified arguments
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

case "$1" in
    training)
        python3 /sdf/home/j/jhirschm/COOKIE_ML/src/ml_backbone/regressions/resnet_phase_regression_training.py
        ;;
    
    evaluation)
        python3 /sdf/home/j/jhirschm/COOKIE_ML/src/ml_backbone/regressions/resnet_phase_regression_evaluation.py
        ;;
    *)
        echo "Invalid script specified. Usage: sbatch this_script.sh [training|evaluation]"
        exit 1
        ;;
esac

echo Finished at: `date`