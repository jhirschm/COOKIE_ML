from dp_utils import *


def plot_ximg_ypdf(h5_file_path, num_examples=3):
    """
    Plot the Ximg and Ypdf datasets and print the attributes for a few example image keys.

    Args:
    - h5_file_path (str): Path to the HDF5 file.
    - num_examples (int): Number of example image keys to plot and print.
    """
    with h5py.File(h5_file_path, 'r') as h5f:
        image_keys = list(h5f.keys())[:num_examples]
        
        for image_key in image_keys:
            print(f"Image Key: {image_key}")
            ximg = h5f[image_key]["Ximg"][:]
            ypdf = h5f[image_key]["Ypdf"][:]
            
            attributes = dict(h5f[image_key].attrs)
            print("Attributes:", attributes)
            
            # Plot Ximg
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title(f"{image_key} - Ximg")
            plt.imshow(ximg, aspect='auto', cmap='viridis')
            plt.colorbar()
            

            plt.subplot(1, 2, 2)
            plt.title(f"{image_key} -Ypdf")
            plt.imshow(ypdf, aspect='auto', cmap='viridis')
            plt.colorbar()
            
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    # Path to the processed HDF5 file
    processed_h5_file_path = "/Users/jhirschm/Documents/MRCO/Data_Changed/Test/2_subpulse_Mar13_2024_1.000_processed.h5"
    
    # Plot and print attributes for a few example image keys
    plot_ximg_ypdf(processed_h5_file_path)
