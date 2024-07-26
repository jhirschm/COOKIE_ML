import os
import tkinter as tk
from tkinter import ttk, messagebox
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Default directory path
default_directory_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/even-dist_Pulses_03302024/Processed_07262024/train/"
# Function to update the file list based on the directory path
def update_file_list(directory_path):
    if not os.path.exists(directory_path):
        messagebox.showerror("Error", f"Directory '{directory_path}' does not exist. Please enter a valid path.")
        return []
    files = [file for file in os.listdir(directory_path) if file.endswith('.h5')]
    return files

# Function to handle changes to the directory path
def change_directory_path(*args):
    global directory_path
    directory_path = directory_path_var.get()
    files = update_file_list(directory_path)
    if files:
        filename_var.set(files[0])
        file_selector['values'] = files
        update_image_keys()
    else:
        filename_var.set("")
        file_selector['values'] = []



# Function to update the available image keys based on the selected file
def update_image_keys(*args):
    selected_file = filename_var.get()
    if not selected_file:
        return
    full_path = os.path.join(directory_path, selected_file)
    with h5py.File(full_path, 'r') as h5_file:
        image_keys = list(h5_file.keys())
    image_key_var.set(image_keys[0])
    image_key_selector['values'] = image_keys
    update_plot()

# Function to update the plot based on the selected file and image key
def update_plot(*args):
    selected_file = filename_var.get()
    selected_key = image_key_var.get()
    full_path = os.path.join(directory_path, selected_file)
    
    with h5py.File(full_path, 'r') as h5_file:
        # ximg = h5_file[selected_key]['Ximg'][:]
        # ypdf = h5_file[selected_key]['Ypdf'][:]
        # # Assuming third_img is also Ypdf for now
        # third_img = ypdf
        ximg = h5_file[selected_key]['Ximg'][:]
        ypdf = h5_file[selected_key]['Ypdf'][:]
        # Assuming third_img is also Ypdf for now
        # third_img = h5_file[selected_key]['output'][:]

    
    # Clear previous plots
    ax1.clear()
    ax2.clear()
    # ax3.clear()

    # Plot Ximg
    im1 = ax1.imshow(ximg, aspect='auto', cmap='viridis')
    ax1.set_title(f"Ximg")
    # if hasattr(update_plot, 'cbar1') and update_plot.cbar1 is not None:
    #     update_plot.cbar1.remove()
    # update_plot.cbar1 = fig.colorbar(im1, ax=ax1)

    # Plot Ypdf
    im2 = ax2.imshow(ypdf, aspect='auto', cmap='viridis')
    ax2.set_title(f"Ypdf")
    # if hasattr(update_plot, 'cbar2') and update_plot.cbar2 is not None:
    #     update_plot.cbar2.remove()
    # update_plot.cbar2 = fig.colorbar(im2, ax=ax2)

    # # Plot the third image (initially set to Ypdf)
    # im3 = ax3.imshow(third_img, aspect='auto', cmap='viridis')
    # ax3.set_title(f"Third Image")
    # # if hasattr(update_plot, 'cbar3') and update_plot.cbar3 is not None:
    # #     update_plot.cbar3.remove()
    # # update_plot.cbar3 = fig.colorbar(im3, ax=ax3)
    
    # Update the canvas with new plots
    canvas.draw()

    # Display the attributes of the selected image key
    with h5py.File(full_path, 'r') as h5_file:
        attributes = h5_file[selected_key].attrs
        
        attributes_text_list = []
        for key in attributes.keys():
            value = attributes[key]
            if key == 'energies':
                value *= 511
            elif key == 'phases':
                value *= 2 * np.pi
            attributes_text_list.append(f"{key}: {value}")
        
        attributes_text = "\n".join(attributes_text_list)
        attributes_label.config(text=attributes_text)
    
    print("End of update")


def analyze_data():
    phase_differences = []
    full_path_list = [os.path.join(directory_path, file) for file in update_file_list(directory_path)]
    
    for full_path in full_path_list:
        with h5py.File(full_path, 'r') as h5_file:
            for key in h5_file.keys():
                if h5_file[key].attrs.get('npulses', 0) == 2:
                    phases = h5_file[key].attrs['phases'] * 2 * np.pi
                    phase_diff = (phases[0] - phases[1])
                    phase_differences.append(phase_diff)
    
    if not phase_differences:
        messagebox.showinfo("Info", "No data points with npulses = 2 found.")
        return

    phase_differences = np.array(phase_differences)
    abs_phase_differences = np.abs(phase_differences)

    sin_phase_differences = np.sin(phase_differences)

    # Plot the histogram of the absolute phase differences
    fig_hist, (ax_hist1, ax_hist2) = plt.subplots(1, 2, figsize=(12, 5))
    ax_hist1.hist(abs_phase_differences, bins=50, color='blue', alpha=0.7)
    ax_hist1.set_title('Histogram of Absolute Phase Differences')
    ax_hist1.set_xlabel('Phase Difference (radians)')
    ax_hist1.set_ylabel('Frequency')

    # Plot the histogram of cos^2(phase_diff) + sin^2(phase_diff)
    ax_hist2.hist(sin_phase_differences, bins=50, color='green', alpha=0.7)
    ax_hist2.set_title('Histogram of sin(phase_diff)')
    ax_hist2.set_xlabel('sin(phase_diff) ')
    ax_hist2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
# Initialize color bar attributes
update_plot.cbar1 = None
update_plot.cbar2 = None
# update_plot.cbar3 = None

# Create the main tkinter window
root = tk.Tk()
root.title("HDF5 File and Example Selector")

# Create a StringVar to hold the selected filename and directory path
filename_var = tk.StringVar()
directory_path_var = tk.StringVar(value=default_directory_path)
image_key_var = tk.StringVar()

# Create a text entry field for directory path
directory_path_entry = tk.Entry(root, textvariable=directory_path_var, width=50)
directory_path_entry.pack(pady=10)
# directory_path_var.trace_add('write', change_directory_path)
directory_path_entry.bind('<Return>', change_directory_path)

# Create a dropdown menu (Combobox) for file selection
file_selector = ttk.Combobox(root, textvariable=filename_var)
file_selector.pack(pady=10)

# Create a dropdown menu (Combobox) for image key selection
image_key_selector = ttk.Combobox(root, textvariable=image_key_var)
image_key_selector.pack(pady=10)

# Create a label to display the attributes
attributes_label = tk.Label(root, text="")
attributes_label.pack(pady=10)

# Create a button to perform the analysis
analyze_button = tk.Button(root, text="Analyze Data", command=analyze_data)
analyze_button.pack(pady=10)
# Create a Matplotlib figure and axes
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,9))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,9))

# Create a canvas to embed the Matplotlib figure
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=10)

# Set the initial directory path and file list
directory_path = default_directory_path
files = update_file_list(directory_path)
if files:
    filename_var.set(files[0])
    file_selector['values'] = files
    update_image_keys()

# Update the plot whenever the image key selection changes
image_key_var.trace_add('write', update_plot)

# Run the tkinter main loop
root.mainloop()
