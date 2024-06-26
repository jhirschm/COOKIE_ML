import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import torch
import torch.nn as nn
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the utils directory relative to the current file's directory
utils_dir = os.path.abspath(os.path.join(current_dir, '..', 'denoising'))
sys.path.append(utils_dir)

from ximg_to_ypdf_autoencoder import Ximg_to_Ypdf_Autoencoder  # Importing the model

# Default directory path
default_directory_path = "/Users/jhirschm/Documents/MRCO/Data_Changed/Test/"

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
model_input_var = tk.StringVar(value="Ximg")  # Option to choose model input
weights_path_var = tk.StringVar()  # Path for model weights

# Initialize autoencoder globally
encoder_layers = [
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
]

decoder_layers = [
    nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
    nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
    nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
]

autoencoder = Ximg_to_Ypdf_Autoencoder(encoder_layers, decoder_layers)

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

# Function to load the model weights
def load_weights():
    file_path = filedialog.askopenfilename()
    weights_path_var.set(file_path)
    autoencoder.load_state_dict(torch.load(file_path))
    messagebox.showinfo("Success", f"Weights loaded from {file_path}")

# Function to update the plot based on the selected file and image key
def update_plot(*args):
    selected_file = filename_var.get()
    selected_key = image_key_var.get()
    full_path = os.path.join(directory_path, selected_file)
    
    with h5py.File(full_path, 'r') as h5_file:
        ximg = h5_file[selected_key]['Ximg'][:]
        ypdf = h5_file[selected_key]['Ypdf'][:]
        # Determine the input for the model
        model_input = ximg if model_input_var.get() == "Ximg" else ypdf
        model_input_tensor = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        model_output = autoencoder(model_input_tensor).squeeze(0).squeeze(0).detach().numpy()
    
    # Clear previous plots
    ax1.clear()
    ax2.clear()
    ax3.clear()

    # Plot Ximg
    im1 = ax1.imshow(ximg, aspect='auto', cmap='viridis')
    ax1.set_title(f"{selected_key} - Ximg")
    if hasattr(update_plot, 'cbar1') and update_plot.cbar1 is not None:
        update_plot.cbar1.remove()
    update_plot.cbar1 = fig.colorbar(im1, ax=ax1)

    # Plot Ypdf
    im2 = ax2.imshow(ypdf, aspect='auto', cmap='viridis')
    ax2.set_title(f"{selected_key} - Ypdf")
    if hasattr(update_plot, 'cbar2') and update_plot.cbar2 is not None:
        update_plot.cbar2.remove()
    update_plot.cbar2 = fig.colorbar(im2, ax=ax2)

    # Plot the model output
    im3 = ax3.imshow(model_output, aspect='auto', cmap='viridis')
    ax3.set_title(f"{selected_key} - Model Output")
    if hasattr(update_plot, 'cbar3') and update_plot.cbar3 is not None:
        update_plot.cbar3.remove()
    update_plot.cbar3 = fig.colorbar(im3, ax=ax3)
    
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
directory_path_var.trace_add('write', change_directory_path)

# Create a dropdown menu (Combobox) for file selection
file_selector = ttk.Combobox(root, textvariable=filename_var)
file_selector.pack(pady=10)

# Create a dropdown menu (Combobox) for image key selection
image_key_selector = ttk.Combobox(root, textvariable=image_key_var)
image_key_selector.pack(pady=10)

# Create a label to display the attributes
attributes_label = tk.Label(root, text="")
attributes_label.pack(pady=10)

# Create a radio button to select model input
input_label = tk.Label(root, text="Model Input:")
input_label.pack(pady=5)
input_ximg = tk.Radiobutton(root, text="Ximg", variable=model_input_var, value="Ximg")
input_ypdf = tk.Radiobutton(root, text="Ypdf", variable=model_input_var, value="Ypdf")
input_ximg.pack()
input_ypdf.pack()

# Create a button to load model weights
load_weights_button = tk.Button(root, text="Load Model Weights", command=load_weights)
load_weights_button.pack(pady=10)

# Create a text entry field for weights path (for display purposes)
weights_path_entry = tk.Entry(root, textvariable=weights_path_var, width=50)
weights_path_entry.pack(pady=10)

# Create a Matplotlib figure and axes
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

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