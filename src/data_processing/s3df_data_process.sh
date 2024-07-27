#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:prjs2e21
#SBATCH --job-name=reg
#SBATCH --output=/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/s3df_runtime_outputs/output-%j.txt
#SBATCH --error=/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/s3df_runtime_outputs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10g
#SBATCH --time=0-10:00:00
##SBATCH --gpus 3
source ~/conda.sh
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
# Define paths to your data files and scalers
FILE_PATHS="/sdf/data/lcls/ds/prj/prjs2e21/results/even-dist_Pulses_03302024/"
SCALER_NAME="min_max_scaler"
SAVEPATH="/sdf/data/lcls/ds/prj/prjs2e21/results/even-dist_Pulses_03302024/Processed_07262024_0to1/"
ENERGY_ELEMENTS=512
TEST_MODE=False

TRAIN=0.9
VAL=0
TEST=0.1

# Run the Python script with the specified arguments
python3 /sdf/home/j/jhirschm/COOKIE_ML/src/data_processing/universal_cookiesimslim_processor.py $FILE_PATHS --scaler_name $SCALER_NAME  --savepath $SAVEPATH --energy_elements $ENERGY_ELEMENTS --train_val_test_split $TRAIN $VAL $TEST 

