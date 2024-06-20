'''
This code serves as universal data processor for CookieSimSlim generated datasets for ML models. By default will retain number of pulses, pulse phases (up to five), pulse energies (up to five), Ximg, and Ypdf. Furthermore, 
for Ximg and Ypdf applies a scaler to normalize the data. Energy is normalized by dividing by 512.

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


    for file_path in file_paths:
        with h5py.File(file_path, 'r') as h5f:
            for image_key in h5f.keys():
                image = h5f[image_key]["Ximg"][:]
                ypd = h5f[image_key]["Ypdf"][:]
                scaler_ximg.partial_fit(image)
                scaler_ypdf.partial_fit(ypd)

    # Save the scaler to a file
    full_scaler_save_path = os.path.join(scaler_save_path, scaler_name)
    joblib.dump(scaler_ximg, full_scaler_save_path+"_ximg.joblib")
    joblib.dump(scaler_ypdf, full_scaler_save_path+"_ypdf.joblib")

    print(f"Scalers saved to {full_scaler_save_path}")


def load_and_preprocess_data(file_paths, ximg_scaler_load_path, ypdf_scaler_load_path, savepath,  energy_elements=512):
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
        save_file_path = os.path.join(savepath, os.path.basename(file_path))
        
        with h5py.File(file_path, 'r') as h5f, h5py.File(save_file_path, 'w') as save_h5f:
            for image_key in h5f.keys():
                npulse = h5f[image_key].attrs["sasecenters"].shape[0]
                energy = h5f[image_key].attrs["sasecenters"]
                phase = h5f[image_key].attrs["sasephases"]
                image = h5f[image_key]["Ximg"][:]
                ypdf = h5f[image_key]["Ypdf"][:]

                min_max_scaled_ximg = min_max_ximg.transform(image)
                min_max_scaled_ypdf = min_max_ypdf.transform(ypdf)

                # Pad or truncate phases and energies to 5 elements
            
                padded_scaled_energy = padded_energy / (energy_elements-1)
                padded_phase = pad_or_truncate(phase, length=5)

                # Save the scaled data and attributes to the new file
                grp = save_h5f.create_group(image_key)
                grp.create_dataset("Ximg", data=min_max_scaled_ximg)
                grp.create_dataset("Ypdf", data=min_max_scaled_ypdf)
                grp.attrs["energies"] = padded_energy
                grp.attrs["phases"] = padded_phase
                grp.attrs["npulses"] = npulse
    

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process CookieSimSlim data for ML models")
    parser.add_argument("file_paths", nargs="+", help="Paths to the data files to process")
    parser.add_argument("--scaler_save_path", default="scalers", help="Path to save the scalers")
    parser.add_argument("--scaler_name", default="min_max_scaler", help="Name to save the scaler as")
    parser.add_argument("--ximg_scaler_load_path", default="scalers/min_max_scaler_ximg.joblib", help="Path to load the Ximg scaler from")
    parser.add_argument("--ypdf_scaler_load_path", default="scalers/min_max_scaler_ypdf.joblib", help="Path to load the Ypdf scaler from")
    parser.add_argument("--savepath", default="processed_data", help="Path to save the processed data")
    parser.add_argument("--energy_elements", default=512, type=int, help="Number of energy elements in the data")

    args = parser.parse_args()

    # Calculate and save the scalers
    calculate_scaler(args.file_paths, args.scaler_save_path, args.scaler_name)

    # Load and preprocess the data
    load_and_preprocess_data(args.file_paths, args.ximg_scaler_load_path, args.ypdf_scaler_load_path, args.savepath, args.energy_elements)

if __name__ == "__main__":
    main()
