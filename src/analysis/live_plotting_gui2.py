import os
import tkinter as tk
from tkinter import ttk, messagebox
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Default directory path
default_directory_path = "/sdf/scratch/lcls/ds/prj/prjs2e21/scratch/fast_data_access/2-Pulse_04232024/Processed_07312024_0to1/train/"

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
    try:
        with h5py.File(full_path, 'r') as h5_file:
            image_keys = list(h5_file.keys())
        if image_keys:
            image_key_var.set(image_keys[0])
            image_key_selector['values'] = image_keys
            update_plot()  # Automatically update the plot on selection
    except OSError as e:
        messagebox.showerror("Error", f"Error opening file '{full_path}': {e}")

# Function to update the plot based on the selected file and key
def update_plot(*args):
    selected_file = filename_var.get()
    selected_key = image_key_var.get()
    full_path = os.path.join(directory_path, selected_file)
    
    try:
        with h5py.File(full_path, 'r') as h5_file:
            # Fetching datasets from the selected key
            if 'Ximg' in h5_file[selected_key]:
                ximg = h5_file[selected_key]['Ximg'][:]
            else:
                messagebox.showerror("Error", f"'Ximg' dataset not found in '{selected_key}'")
                return
            
            if 'Ypdf' in h5_file[selected_key]:
                ypdf = h5_file[selected_key]['Ypdf'][:]
            else:
                messagebox.showerror("Error", f"'Ypdf' dataset not found in '{selected_key}'")
                return
            
            third_img = ypdf  # Example; replace if you have a third dataset
            
            # Attributes extraction
            attributes = h5_file[selected_key].attrs
            energies = np.round(attributes['energies'], 3)
            phases = np.round(attributes['phases'], 3)
            npulses = attributes['npulses']
    except OSError as e:
        messagebox.showerror("Error", f"Error reading file '{full_path}': {e}")
        return
    except KeyError as e:
        messagebox.showerror("Error", f"KeyError: {e}")
        return

    # Clear previous plots
    ax1.clear()
    ax2.clear()
    ax3.clear()

    # Plot Ximg
    im1 = ax1.imshow(ximg, aspect='auto', cmap='viridis')
    ax1.set_title(f"Ximg")

    # Plot Ypdf
    im2 = ax2.imshow(ypdf, aspect='auto', cmap='viridis')
    ax2.set_title(f"Ypdf")

    # Plot the third image (initially set to Ypdf)
    im3 = ax3.imshow(third_img, aspect='auto', cmap='viridis')
    ax3.set_title(f"Third Image")

    # Annotate with attributes
    ax1.text(0.05, 0.95, f"Energies: {energies}", transform=ax1.transAxes, fontsize=10, color='white', verticalalignment='top')
    ax2.text(0.05, 0.95, f"Phases: {phases}", transform=ax2.transAxes, fontsize=10, color='white', verticalalignment='top')
    ax3.text(0.05, 0.95, f"Num Pulses: {npulses}", transform=ax3.transAxes, fontsize=10, color='white', verticalalignment='top')

    # Update the canvas with new plots
    canvas.draw()

# Create the main tkinter window
root = tk.Tk()
root.title("HDF5 File and Example Selector")

# Create StringVar to hold the selected filename, directory path, and image key
filename_var = tk.StringVar()
directory_path_var = tk.StringVar(value=default_directory_path)
image_key_var = tk.StringVar()

# Create a text entry field for directory path
directory_path_entry = tk.Entry(root, textvariable=directory_path_var, width=50)
directory_path_entry.pack(pady=10)
directory_path_entry.bind('<Return>', change_directory_path)

# Create a dropdown menu (Combobox) for file selection
file_selector = ttk.Combobox(root, textvariable=filename_var)
file_selector.pack(pady=10)

# Create a dropdown menu (Combobox) for image key selection
image_key_selector = ttk.Combobox(root, textvariable=image_key_var)
image_key_selector.pack(pady=10)

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
