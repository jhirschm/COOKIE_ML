#!/bin/bash

# Define paths to your data files and scalers
FILE_PATHS="/Users/jhirschm/Documents/MRCO/Data_Changed/2_Subpulse_LocalRun_Mar13-2024"
SCALER_SAVE_PATH="/Users/jhirschm/Documents/MRCO/Data_Changed/"
SCALER_NAME="test_min_max_scaler"
SAVEPATH="/Users/jhirschm/Documents/MRCO/Data_Changed/Test"
ENERGY_ELEMENTS=512

# Run the Python script with the specified arguments
python /Users/jhirschm/Documents/MRCO/COOKIE_ML/src/data_processing/universal_cookiesimslim_processor.py $FILE_PATHS --scaler_save_path $SCALER_SAVE_PATH --scaler_name $SCALER_NAME  --savepath $SAVEPATH --energy_elements $ENERGY_ELEMENTS