import os
import tkinter as tk
from tkinter import ttk, messagebox
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Default directory path
default_directory_path = "/Users/jhirschm/Documents/MRCO/Data_Changed/Test/"
default_directory_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_06252024_subset4/outputs/"
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

# Create the main tkinter window
root = tk.Tk()
root.title("HDF5 File and Example Selector")

# Create a StringVar to hold the selected filename and directory path
filename_var = tk.StringVar()
directory_path_var = tk.StringVar(value=default_directory_path)
image_key_var = tk.StringVar()

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
        ximg = h5_file[selected_key]['input'][:]
        ximg = np.reshape(ximg, (32, 16, 512))
        ypdf = h5_file[selected_key]['target'][:]
        ypdf = np.reshape(ypdf, (32, 16, 512))
        # Assuming third_img is also Ypdf for now
        third_img = h5_file[selected_key]['output'][:]
        third_img = np.reshape(third_img, (32, 16, 512))

    
    # Clear previous plots
    ax1.clear()
    ax2.clear()
    ax3.clear()

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

    # Plot the third image (initially set to Ypdf)
    im3 = ax3.imshow(third_img, aspect='auto', cmap='viridis')
    ax3.set_title(f"Third Image")
    # if hasattr(update_plot, 'cbar3') and update_plot.cbar3 is not None:
    #     update_plot.cbar3.remove()
    # update_plot.cbar3 = fig.colorbar(im3, ax=ax3)
    
    # Update the canvas with new plots
    canvas.draw()

    # Display the attributes of the selected image key
    with h5py.File(full_path, 'r') as h5_file:
        attributes = h5_file[selected_key].attrs
        attributes_text = "\n".join([f"{key}: {attributes[key]}" for key in attributes.keys()])
        attributes_label.config(text=attributes_text)
    
    print("End of update")

# Initialize color bar attributes
update_plot.cbar1 = None
update_plot.cbar2 = None
update_plot.cbar3 = None

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

# Create a Matplotlib figure and axes
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,9))

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
