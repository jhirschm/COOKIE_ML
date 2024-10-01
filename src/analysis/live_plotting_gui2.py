import os
import tkinter as tk
from tkinter import ttk, messagebox
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Default directory path
default_directory_path = "/Users/jhirschm/Documents/MRCO/Data_Changed/Test/"
default_directory_path = "/sdf/scratch/lcls/ds/prj/prjs2e21/scratch/fast_data_access/2-Pulse_04232024/Processed_07312024_0to1/train/"
# Function to update the file list based on the directory path

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

def update_plot(*args):
    selected_file = filename_var.get()
    selected_key = image_key_var.get()
    full_path = os.path.join(directory_path, selected_file)
    
    with h5py.File(full_path, 'r') as h5_file:
        ximg = h5_file[selected_key]['Ximg'][:]
        ypdf = h5_file[selected_key]['Ypdf'][:]
        # Assuming third_img is also Ypdf for now
        third_img = ypdf#h5_file[selected_key]['output'][:]

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
    
    # Update the canvas with new plots
    canvas.draw()

    # Display the attributes of the selected image key
    with h5py.File(full_path, 'r') as h5_file:
        attributes = h5_file[selected_key].attrs
        energies = np.round(attributes['energies'], 3)  # Round energy array to 3 decimal places
        phases = np.round(attributes['phases'], 3)  # Round phases array to 3 decimal places
        npulses = attributes['npulses']  # Number of pulses (fixed at 2)
        
        # Create a string to display the attributes
        attributes_text = f"Energies: {energies}\nPhases: {phases}\nNumber of Pulses: {npulses}"
        
        # Update the label with attributes
        attributes_label.config(text=attributes_text)

    # Annotate the values on each subplot
    ax1.text(0.05, 0.95, f"Energies: {energies}", transform=ax1.transAxes, fontsize=10, color='white', verticalalignment='top')
    ax1.text(0.05, 0.85, f"Phases: {phases}", transform=ax1.transAxes, fontsize=10, color='white', verticalalignment='top')
    ax1.text(0.05, 0.75, f"Num Pulses: {npulses}", transform=ax1.transAxes, fontsize=10, color='white', verticalalignment='top')

    ax2.text(0.05, 0.95, f"Energies: {energies}", transform=ax2.transAxes, fontsize=10, color='white', verticalalignment='top')
    ax2.text(0.05, 0.85, f"Phases: {phases}", transform=ax2.transAxes, fontsize=10, color='white', verticalalignment='top')
    ax2.text(0.05, 0.75, f"Num Pulses: {npulses}", transform=ax2.transAxes, fontsize=10, color='white', verticalalignment='top')

    ax3.text(0.05, 0.95, f"Energies: {energies}", transform=ax3.transAxes, fontsize=10, color='white', verticalalignment='top')
    ax3.text(0.05, 0.85, f"Phases: {phases}", transform=ax3.transAxes, fontsize=10, color='white', verticalalignment='top')
    ax3.text(0.05, 0.75, f"Num Pulses: {npulses}", transform=ax3.transAxes, fontsize=10, color='white', verticalalignment='top')

    print(f"Energies: {energies}")
    print(f"Phases: {phases}")
    print(f"Number of Pulses: {npulses}")

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
