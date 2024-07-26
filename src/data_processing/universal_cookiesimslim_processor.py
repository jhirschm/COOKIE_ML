'''
This code serves as universal data processor for CookieSimSlim generated datasets for ML models. By default will retain number of pulses, pulse phases (up to five), pulse energies (up to five), Ximg, and Ypdf. Furthermore, 
for Ximg and Ypdf applies a scaler to normalize the data. Energy is normalized by dividing by 512 and phase by dividing by 2*pi. The processed data is saved in a new file with the default suffix "_processed" appended to the original filename.

'''
from dp_utils import *
def calculate_scaler(file_paths, scaler_save_path, scaler_name):
    """
    Calculate and save a MinMaxScaler based on the data in the specified files.

    Args:
    - file_paths (list): List of file paths containing data for scaling.
    - scaler_save_path (str): Path where the scaler should be saved.

    Returns:
    - scaler (MinMaxScaler): The trained MinMaxScaler.
    """
    scaler_ximg = MinMaxScaler(feature_range=(-1, 1))
    scaler_ypdf = MinMaxScaler(feature_range=(-1, 1))

    # Check if savepath exists, and create it if it doesn't
    if not os.path.exists(scaler_save_path):
        os.makedirs(scaler_save_path)
        print(f"Directory {scaler_save_path} created.")

    for file_path in file_paths:
        with h5py.File(file_path, 'r') as h5f:
            for image_key in h5f.keys():
                image = h5f[image_key]["Ximg"][:]
                ypd = h5f[image_key]["Ypdf"][:]
                scaler_ximg.partial_fit(image)
                scaler_ypdf.partial_fit(ypd)

    # Save the scaler to a file
    full_scaler_save_path = os.path.join(scaler_save_path, scaler_name)
    ximg_scaler_save_path = full_scaler_save_path + "_ximg.joblib"
    ypdf_scaler_save_path = full_scaler_save_path + "_ypdf.joblib"
    joblib.dump(scaler_ximg, ximg_scaler_save_path)
    joblib.dump(scaler_ypdf, ypdf_scaler_save_path)

    print(f"Scalers saved to {full_scaler_save_path}")

    return ximg_scaler_save_path, ypdf_scaler_save_path


def load_and_preprocess_data(file_paths, ximg_scaler_load_path, ypdf_scaler_load_path, savepath,  energy_elements=512, suffix="_processed"):
    """
    Load and preprocess the data from the specified files, saving the processed data to a new file.
    """
    # Load the scalers
    min_max_ximg = joblib.load(ximg_scaler_load_path)
    min_max_ypdf = joblib.load(ypdf_scaler_load_path)

    # Check if savepath exists, and create it if it doesn't
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        print(f"Directory {savepath} created.")
    
    
    # Process each file
    for file_path in file_paths:
        base_filename = os.path.basename(file_path)
        save_filename = f"{os.path.splitext(base_filename)[0]}{suffix}.h5"
        save_file_path = os.path.join(savepath, save_filename)
        
        with h5py.File(file_path, 'r') as h5f, h5py.File(save_file_path, 'w') as save_h5f:
            for image_key in h5f.keys():
                npulse = h5f[image_key].attrs["sasecenters"].shape[0]
                energy = h5f[image_key].attrs["sasecenters"]
                energy = np.asarray(energy)
                phase = h5f[image_key].attrs["sasephases"]
                phase = np.asarray(phase)
                image = h5f[image_key]["Ximg"][:]
                ypdf = h5f[image_key]["Ypdf"][:]

                min_max_scaled_ximg = min_max_ximg.transform(image)
                min_max_scaled_ypdf = min_max_ypdf.transform(ypdf)

            
                scaled_energy = energy / (energy_elements-1)
                scaled_phase = phase / (2*np.pi)
                
                # Save the scaled data and attributes to the new file
                grp = save_h5f.create_group(image_key)
                grp.create_dataset("Ximg", data=min_max_scaled_ximg)
                grp.create_dataset("Ypdf", data=min_max_scaled_ypdf)
                grp.attrs["energies"] = scaled_energy
                grp.attrs["phases"] = scaled_phase
                grp.attrs["npulses"] = npulse
    

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process CookieSimSlim data for ML models")
    parser.add_argument("file_paths", type = str, help="Paths to the data files to process")
    # parser.add_argument("--scaler_save_path", default="scalers", help="Path to save the scalers")
    parser.add_argument("--scaler_name", default="min_max_scaler", help="Name to save the scaler as")
    parser.add_argument("--ximg_scaler_load_path", default="scalers/min_max_scaler_ximg.joblib", help="Path to load the Ximg scaler from")
    parser.add_argument("--ypdf_scaler_load_path", default="scalers/min_max_scaler_ypdf.joblib", help="Path to load the Ypdf scaler from")
    parser.add_argument("--savepath", default="processed_data", help="Path to save the processed data")
    parser.add_argument("--energy_elements", default=512, type=int, help="Number of energy elements in the data")
    parser.add_argument("--suffix", default="_processed", help="Suffix to append to the processed data filename")
    parser.add_argument("--test_mode", default=False, help="Run in test mode")
    parser.add_argument("--train_val_test_split", type=float, nargs=3, default=[1.0, 0, 0], help="Train, validation, and test split")
    args = parser.parse_args()

    data_file_paths = [os.path.join(args.file_paths, file) for file in os.listdir(args.file_paths) if file.endswith('.h5')]
    if (args.test_mode == True):
        data_file_paths = data_file_paths[0:1]
    print("Data File Paths:")
    print(data_file_paths)
    # Based on train_val_test_split, split the data into training, validation, and test sets and make subfolders in savepath for these
    assert sum(args.train_val_test_split) == 1.0, "Split ratios must sum to 1."

    # Calculate the number of samples for each set
    train_ratio, val_ratio, test_ratio = args.train_val_test_split
    train_files, temp_files = train_test_split(data_file_paths, test_size=(val_ratio + test_ratio))
    val_files, test_files = train_test_split(temp_files, test_size=(test_ratio / (val_ratio + test_ratio)))

    # Create subfolders for train, val, and test
    train_folder = os.path.join(args.savepath, 'train')
    val_folder = os.path.join(args.savepath, 'val')
    test_folder = os.path.join(args.savepath, 'test')
    # Calculate and save the scalers
    if train_files != []:
        ximg_scaler_load_path, ypdf_scaler_load_path = calculate_scaler(train_files, train_folder, args.scaler_name)
        load_and_preprocess_data(train_files, ximg_scaler_load_path, ypdf_scaler_load_path, train_folder, args.energy_elements, args.suffix)
    if val_files != []:
        ximg_scaler_load_path, ypdf_scaler_load_path = calculate_scaler(data_file_paths, val_folder, args.scaler_name)
        load_and_preprocess_data(val_files, ximg_scaler_load_path, ypdf_scaler_load_path, val_folder, args.energy_elements, args.suffix)
    if test_files != []:
        ximg_scaler_load_path, ypdf_scaler_load_path = calculate_scaler(data_file_paths, test_folder, args.scaler_name)
        load_and_preprocess_data(test_files, ximg_scaler_load_path, ypdf_scaler_load_path, test_folder, args.energy_elements, args.suffix)

    # # Load and preprocess the data
    # load_and_preprocess_data(data_file_paths, ximg_scaler_load_path, ypdf_scaler_load_path, args.savepath, args.energy_elements, args.suffix)

if __name__ == "__main__":
    main()
