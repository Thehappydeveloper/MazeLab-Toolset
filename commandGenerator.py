"""
Point Cloud Generator GUI

This program provides a graphical user interface (GUI) for generating point cloud data commands
based on user-selected experiments and participants. The application allows users to:
1. Browse and select dataset and output folders.
2. Configure various settings such as frames per second (FPS), material color, light color, float precision, sphere density, and prism density.
3. Select experiments and participants from the dataset.
4. Generate and copy a command string to the clipboard for point cloud data generation.

The generated command string includes the selected settings and the experiments/participants, and it can be used to run an external program (e.g., 'myprogram') with the specified parameters.

The main components of the program include:
- Functions to handle folder browsing, updating the treeviews, generating the command string, and user interactions.
- Tkinter widgets for user inputs and displaying the data.

Dependencies:
- tkinter
- os
- re
"""


import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import re

from collections import defaultdict
import os
import json
import base64

def generate_command():
    """
    Generates a command string based on user-selected experiments and participants,
    as well as additional settings. The command is copied to the clipboard and displayed
    in the GUI.

    The command is formatted as:
    python myprogram --input_path {input_path} --output_path {output_path}
    --fps {fps_value} --material_color {material_color_value} 
    --light_color {light_color_value} --float_precision {float_precision_value} 
    --sphere_density {sphere_density_value} --prism_density {prism_density_value} 
    --sphere_inclusion {sphere_inclusion_value} 
    --prisms_inclusion {prisms_inclusion_value} --point_clouds_inclusion {point_clouds_inclusion_value}
    --normalize_point_cloud {normalize_value}
    --ply_format {ply_format_value} --pcds_point_cap {pcds_point_cap_value}
    --experiments {experiment_dict}

    It includes all selected experiments and participants.
    """
    selected_pairs = [(selected_tree.item(item, "text").split(' - ')[0], selected_tree.item(item, "text").split(' - ')[1]) for item in selected_tree.get_children()]

    experiment_dict = defaultdict(list)
    for experiment, participant in selected_pairs:
        experiment_dict[experiment].append(participant)

    # Convert defaultdict to a regular dict for JSON serialization
    experiment_dict = dict(experiment_dict)

    fps_value = fps_combobox.get()
    material_color_value = material_color_combobox.get()
    light_color_value = light_color_combobox.get()
    float_precision_value = float_precision_combobox.get()
    sphere_density_value = sphere_density_combobox.get()
    prism_density_value = prism_density_combobox.get()
    sphere_inclusion_value = sphere_inclusion_combobox.get()
    prisms_inclusion_value = prisms_inclusion_combobox.get()
    point_clouds_inclusion_value = point_clouds_inclusion_combobox.get()
    ply_format_value = ply_format_combobox.get()
    pcds_point_cap_value = pcds_point_cap_combobox.get()
    normalize_value = normalize_combobox.get()  # New parameter for normalization

    input_path = dataset_folder_entry.get()
    output_path = output_folder_entry.get()

    # Adjust input_path format based on OS
    if os.name == 'nt':  # Windows
        input_path = input_path.replace('/', '\\')
    else:  # Linux
        input_path = input_path.replace('\\', '/')

    if not output_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "pcdDataset")

    # Convert experiment_dict to a JSON string and then to a base64-encoded string
    experiment_dict_str = json.dumps(experiment_dict)
    experiment_dict_base64 = base64.b64encode(experiment_dict_str.encode('utf-8')).decode('utf-8')

    command = (
        f"python main.py --input_path \"{input_path}\" --output_path \"{output_path}\" "
        f"--fps {fps_value} --material_color \"{material_color_value}\" "
        f"--light_color {light_color_value} --float_precision {float_precision_value} "
        f"--sphere_density {sphere_density_value} --prism_density {prism_density_value} "
        f"--sphere_inclusion {sphere_inclusion_value} "
        f"--prisms_inclusion {prisms_inclusion_value} --point_clouds_inclusion {point_clouds_inclusion_value} "
        f"--normalize_point_cloud {normalize_value} --ply_format {ply_format_value} --pcds_point_cap {pcds_point_cap_value} "  # Add the normalization parameter
        f"--experiments {experiment_dict_base64}"
    )

    root.clipboard_clear()
    root.clipboard_append(command)
    command_label.config(text="Command copied to the clipboard!", fg="red")
    root.after(2000, clear_label)


def clear_label():
    """
    Clears the text of the command label.
    """
    command_label.config(text="", fg="black")

def get_experiments_and_participants(dataset_folder):
    """
    Scans the dataset folder to retrieve a dictionary of experiments and their participants.

    Args:
        dataset_folder (str): Path to the dataset folder.

    Returns:
        dict: A dictionary where keys are experiment names and values are lists of participants.
    """
    experiments = {}
    experiment_folder = os.path.join(dataset_folder, 'Experiments')
    for experiment in os.listdir(experiment_folder):
        experiment_path = os.path.join(experiment_folder, experiment)
        if os.path.isdir(experiment_path):
            participants = []
            for participant in os.listdir(experiment_path):
                participant_path = os.path.join(experiment_path, participant)
                if os.path.isdir(participant_path):
                    participants.append(participant)
            experiments[experiment] = participants
    return experiments

def extract_experiment_number(experiment_name):
    """
    Extracts the experiment number from the experiment name.

    Args:
        experiment_name (str): The name of the experiment.

    Returns:
        int: The extracted experiment number, or infinity if no number is found.
    """
    match = re.search(r"_(\d+)$", experiment_name)
    return int(match.group(1)) if match else float('inf')

def browse_dataset_folder():
    """
    Opens a file dialog for the user to select the dataset folder.
    Updates the dataset folder entry and experiments treeview.
    """
    folder_path = filedialog.askdirectory()
    if folder_path:
        dataset_folder_entry.delete(0, tk.END)
        dataset_folder_entry.insert(tk.END, folder_path)
        update_experiments()

def browse_output_folder():
    """
    Opens a file dialog for the user to select the output folder.
    Updates the output folder entry.
    """
    folder_path = filedialog.askdirectory()
    if folder_path:
        output_folder_entry.delete(0, tk.END)
        output_folder_entry.insert(tk.END, folder_path)

def select_items():
    """
    Selects items from the experiments treeview and adds them to the selected treeview.
    """
    selected_items = experiments_tree.selection()
    selected_pairs = set()

    for item in selected_items:
        item_text = experiments_tree.item(item, "text")
        parent_item = experiments_tree.parent(item)
        if parent_item:
            parent_text = experiments_tree.item(parent_item, "text")
            selected_pairs.add((parent_text, item_text))
        elif item_text in experiments:
            for participant in experiments[item_text]:
                selected_pairs.add((item_text, participant))

    update_selected_treeview(selected_pairs)

def update_selected_treeview(selected_pairs):
    """
    Updates the selected treeview with the selected pairs.

    Args:
        selected_pairs (set): A set of tuples containing selected experiment and participant pairs.
    """
    selected_pairs = sorted(set(selected_pairs))
    for child in selected_tree.get_children():
        selected_tree.delete(child)
    for experiment, participant in selected_pairs:
        selected_tree.insert("", "end", text=f"{experiment} - {participant}")

def update_experiments():
    """
    Updates the experiments treeview based on the selected dataset folder.
    """
    dataset_folder = dataset_folder_entry.get()
    if os.path.exists(dataset_folder):
        global experiments
        experiments = get_experiments_and_participants(dataset_folder)
        for child in experiments_tree.get_children():
            experiments_tree.delete(child)
        for experiment in sorted(experiments.keys(), key=extract_experiment_number):
            parent_id = experiments_tree.insert("", "end", text=experiment)
            for participant in experiments[experiment]:
                experiments_tree.insert(parent_id, "end", text=participant)
    else:
        messagebox.showerror("Error", "Invalid dataset folder.")

# Create the main window
root = tk.Tk()
root.title("Point Cloud Generator")

# Create a frame for the dataset folder
dataset_frame = tk.Frame(root)
dataset_frame.grid(row=0, column=0, sticky="w", pady=5)

# Dataset folder label
dataset_folder_label = tk.Label(dataset_frame, text="Dataset Folder:")
dataset_folder_label.grid(row=0, column=0, sticky="w")

# Dataset folder entry
dataset_folder_entry = tk.Entry(dataset_frame, width=50)
dataset_folder_entry.grid(row=0, column=1, sticky="we", padx=(0, 5))

# Browse dataset button
browse_dataset_button = tk.Button(dataset_frame, text="Browse", command=browse_dataset_folder)
browse_dataset_button.grid(row=0, column=2, sticky="w", pady=5)

# Create a frame for the output folder
output_frame = tk.Frame(root)
output_frame.grid(row=1, column=0, sticky="w", pady=5)

# Output folder label
output_folder_label = tk.Label(output_frame, text="Output Folder:")
output_folder_label.grid(row=0, column=0, sticky="w")

# Output folder entry
output_folder_entry = tk.Entry(output_frame, width=50)
output_folder_entry.grid(row=0, column=1, sticky="we", padx=(0, 5))

# Browse output button
browse_output_button = tk.Button(output_frame, text="Browse", command=browse_output_folder)
browse_output_button.grid(row=0, column=2, sticky="w", pady=5)

# Comboboxes for additional settings
settings_frame = tk.Frame(root)
settings_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

fps_label = tk.Label(settings_frame, text="FPS:")
fps_label.grid(row=0, column=0, sticky="e")

fps_combobox = ttk.Combobox(settings_frame, values=[1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60], state="readonly")
fps_combobox.grid(row=0, column=1)
fps_combobox.current(6)

material_color_label = tk.Label(settings_frame, text="Material Color:")
material_color_label.grid(row=1, column=0, sticky="e")

material_color_combobox = ttk.Combobox(settings_frame, values=["RGBA", "RGB", "Grey scale", "None"], state="readonly")
material_color_combobox.grid(row=1, column=1)
material_color_combobox.current(1)

light_color_label = tk.Label(settings_frame, text="Light Color:")
light_color_label.grid(row=2, column=0, sticky="e")

light_color_combobox = ttk.Combobox(settings_frame, values=["RGBAI", "RGBA", "RGBI", "RGB", "Intensity", "None"], state="readonly")
light_color_combobox.grid(row=2, column=1)
light_color_combobox.current(2)

float_precision_label = tk.Label(settings_frame, text="Float Precision:")
float_precision_label.grid(row=3, column=0, sticky="e")

float_precision_combobox = ttk.Combobox(settings_frame, values=[16, 32, 64], state="readonly")
float_precision_combobox.grid(row=3, column=1)
float_precision_combobox.current(1)

# Sphere density combobox
sphere_density_label = tk.Label(settings_frame, text="Sphere Density:")
sphere_density_label.grid(row=4, column=0, sticky="e")

sphere_density_combobox = ttk.Combobox(settings_frame, values=["0.8", "0.16"])
sphere_density_combobox.grid(row=4, column=1)
sphere_density_combobox.current(0)

# Prism density combobox
prism_density_label = tk.Label(settings_frame, text="Prism Density:")
prism_density_label.grid(row=5, column=0, sticky="e")

prism_density_combobox = ttk.Combobox(settings_frame, values=["0.08", "0.01"])
prism_density_combobox.grid(row=5, column=1)
prism_density_combobox.current(0)

# Sphere inclusion combobox
sphere_inclusion_label = tk.Label(settings_frame, text="Include Sphere:")
sphere_inclusion_label.grid(row=7, column=0, sticky="e")

sphere_inclusion_combobox = ttk.Combobox(settings_frame, values=["Yes", "No"], state="readonly")
sphere_inclusion_combobox.grid(row=7, column=1)
sphere_inclusion_combobox.current(0)

# Rectangular prisms inclusion combobox
prisms_inclusion_label = tk.Label(settings_frame, text="Include Rectangular Prisms:")
prisms_inclusion_label.grid(row=8, column=0, sticky="e")

prisms_inclusion_combobox = ttk.Combobox(settings_frame, values=["Yes", "No"], state="readonly")
prisms_inclusion_combobox.grid(row=8, column=1)
prisms_inclusion_combobox.current(0)

# Point clouds inclusion combobox
point_clouds_inclusion_label = tk.Label(settings_frame, text="Include Point Clouds:")
point_clouds_inclusion_label.grid(row=9, column=0, sticky="e")

point_clouds_inclusion_combobox = ttk.Combobox(settings_frame, values=["Yes", "No"], state="readonly")
point_clouds_inclusion_combobox.grid(row=9, column=1)
point_clouds_inclusion_combobox.current(0)

# PLY format combobox
ply_format_label = tk.Label(settings_frame, text="PLY Format:")
ply_format_label.grid(row=10, column=0, sticky="e")

ply_format_combobox = ttk.Combobox(settings_frame, values=["Binary", "ASCII"], state="readonly")
ply_format_combobox.grid(row=10, column=1)
ply_format_combobox.current(0)

# PCDs point cap combobox (editable)
pcds_point_cap_label = tk.Label(settings_frame, text="PCDs Point Cap:")
pcds_point_cap_label.grid(row=11, column=0, sticky="e")

pcds_point_cap_combobox = ttk.Combobox(settings_frame, values=[100000, 500000, 1000000, 2000000, 5000000], state="normal")
pcds_point_cap_combobox.grid(row=11, column=1)
pcds_point_cap_combobox.current(0)

# Add normalization combobox
normalize_label = tk.Label(settings_frame, text="Normalize Point Cloud:")
normalize_label.grid(row=12, column=0, sticky="e")

normalize_combobox = ttk.Combobox(settings_frame, values=["Yes", "No"], state="readonly")
normalize_combobox.grid(row=12, column=1)
normalize_combobox.current(1)  # Default value set to "No"

# Title for experiments treeview
experiments_tree_label = tk.Label(root, text="Experiments/Participants:")
experiments_tree_label.grid(row=3, column=0, columnspan=3)

# Experiments treeview with scrollbar
experiments_tree_frame = tk.Frame(root)
experiments_tree_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

experiments_tree_scrollbar = tk.Scrollbar(experiments_tree_frame)
experiments_tree_scrollbar.pack(side="right", fill="y")

experiments_tree = ttk.Treeview(experiments_tree_frame, selectmode="extended", show="tree", yscrollcommand=experiments_tree_scrollbar.set)
experiments_tree.pack(fill="both", expand=True)
experiments_tree_scrollbar.config(command=experiments_tree.yview)

# Selected experiments/participants treeview
selected_tree_label = tk.Label(root, text="Selected Experiments/Participants:")
selected_tree_label.grid(row=5, column=0, columnspan=3)

selected_tree_frame = tk.Frame(root)
selected_tree_frame.grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

selected_tree_scrollbar = tk.Scrollbar(selected_tree_frame)
selected_tree_scrollbar.pack(side="right", fill="y")

selected_tree = ttk.Treeview(selected_tree_frame, show="tree", yscrollcommand=selected_tree_scrollbar.set)
selected_tree.pack(fill="both", expand=True)
selected_tree_scrollbar.config(command=selected_tree.yview)

# Create a frame for the centered label
label_frame = tk.Frame(root)
label_frame.grid(row=7, column=0, columnspan=2, padx=10, pady=10)

# Command Label
command_label = tk.Label(label_frame, text="", fg="red", width=27)
command_label.grid(row=0, column=0, padx=(0, 10), pady=10, sticky="e")

# Create a frame for the buttons
button_frame = tk.Frame(root)
button_frame.grid(row=8, column=0, columnspan=2, padx=10, pady=10)

# Select Button
select_button = tk.Button(button_frame, text="Select Scope", command=select_items)
select_button.grid(row=0, column=0, padx=(0, 5), pady=10, sticky="e")

# Generate Command Button
generate_command_button = tk.Button(button_frame, text="Generate Command", command=generate_command)
generate_command_button.grid(row=0, column=1, padx=(5, 0), pady=10, sticky="w")

# Update experiments on folder selection
dataset_folder_entry.bind("<FocusOut>", lambda event: update_experiments())

root.mainloop()
