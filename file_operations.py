import os
import numpy as np
import re
import sys
import math
from tqdm import tqdm

def parse_rendering_states_from_frame(filepath):
    """
    Parses the rendering states of actors from a given frame file.

    Args:
        filepath (str): Path to the frame file.

    Returns:
        dict: A dictionary with actor names as keys and rendering states (1 for Yes, 0 for No) as values.
    """
    with open(filepath, 'r') as file:
        content = file.read()
    lines = content.splitlines()
    actors = {}
    current_actor = None

    for line in lines:
        if line.endswith('):'):
            parts = line.split()
            current_actor = parts[0]
        elif current_actor and "Rendered:" in line:
            rendered = line.split()[-1]
            actors[current_actor] = 1 if rendered == 'Yes' else 0
            current_actor = None

    return actors


def generate_dynamic_rendering_dict(base_path, experiment_dict):
    """
    Processes the dataset to extract and transform rendering states for specified experiments and participants,
    given a base path and a dictionary of experiments and participants to include.

    Args:
        base_path (str): Path to the base dataset directory.
        experiment_dict (dict): A dictionary where keys are experiment names and values are lists of participant names.

    Returns:
        dict: A nested dictionary with the structure {Experiment: {Participant: {Actor: [rendering state counts]}}}.
    """
    data = {}
    
    # Assuming the structure is base_path/Experiments
    experiments_path = os.path.join(base_path, 'Experiments')

    for experiment, participants_list in tqdm(experiment_dict.items(), desc="Processing Experiments"):
        experiment_path = os.path.join(experiments_path, experiment)
        
        if not os.path.isdir(experiment_path):
            print(f"Experiment directory not found: {experiment_path}")
            continue
        
        participants_data = {}

        for participant in tqdm(participants_list, desc=f"Processing Participants in {experiment}", leave=False):
            participant_path = os.path.join(experiment_path, participant)

            if not os.path.isdir(participant_path):
                print(f"Participant directory not found: {participant_path}")
                continue

            dynamic_actors_path = os.path.join(participant_path, 'DynamicActors')
            if not os.path.isdir(dynamic_actors_path):
                continue

            actor_frames = {}
            frame_files = [frame for frame in os.listdir(dynamic_actors_path) if frame.startswith('frame_') and frame.endswith('.txt')]
            frame_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))  # Sort frame files by their frame number

            for frame_file in tqdm(frame_files, desc=f"Processing Frames in {participant}", leave=False):
                frame_path = os.path.join(dynamic_actors_path, frame_file)

                frame_data = parse_rendering_states_from_frame(frame_path)
                for actor, rendered in frame_data.items():
                    if actor not in actor_frames:
                        actor_frames[actor] = []
                    actor_frames[actor].append(rendered)

            # Transform rendering states to counts and handle transitions from Yes to No
            transformed_actor_frames = {}
            for actor, bool_list in actor_frames.items():
                if isinstance(bool_list, list):
                    continuous_true_count = 0
                    for idx, value in enumerate(bool_list):
                        if value:
                            continuous_true_count += 1
                            bool_list[idx] = continuous_true_count
                        else:
                            continuous_true_count = math.floor(bool_list[idx-1] / 2) if idx > 0 else 0
                            bool_list[idx] = continuous_true_count

                    transformed_actor_frames[actor] = bool_list

            participants_data[participant] = transformed_actor_frames
        data[experiment] = participants_data

    return data

def fetch_visibility_score(path, data_dict):
    """
    Fetches the visibility scores for actors from the dictionary based on the file path.

    Args:
        path (str): The path to the frame file.
        data_dict (dict): The dictionary with the structure {Experiment: {Participant: {Actor: [rendering state counts]}}}.

    Returns:
        dict or None: The visibility scores for the frame if found, otherwise None.
    """
    # Extract parts from the path
    path_parts = path.split(os.sep)
    
    try:
        # Get experiment, participant, and frame number from the path
        experiment = path_parts[path_parts.index('Experiments') + 1]
        participant = path_parts[path_parts.index('Experiments') + 2]
        frame_file = path_parts[-1]
        
        # Extract frame number from the file name
        frame_number = int(re.findall(r'\d+', frame_file)[0])
        
        # Validate the experiment and participant existence
        if experiment in data_dict and participant in data_dict[experiment]:
            actors_rendering_states = data_dict[experiment][participant]
            
            # Prepare result dictionary
            frame_data = {}
            
            for actor, counts in actors_rendering_states.items():
                if frame_number - 1 < len(counts):
                    frame_data[actor] = counts[frame_number - 1]
                else:
                    frame_data[actor] = None
            
            return frame_data
        
        return None
    
    except (IndexError, ValueError, KeyError, TypeError):
        return None

def load_point_cloud(file_path):
    """
    Load point cloud data from a file.

    Parameters:
    file_path (str): Path to the point cloud file.

    Returns:
    numpy.ndarray: Array of point cloud data.
    """
    points = []
    with open(file_path, 'r') as file:
        num_points = int(file.readline().strip())
        for line in file:
            point = [float(x) for x in line.split(',')]
            points.append(point)
    return np.array(points)

def generate_output_file_name(input_file_path):
    """
    Generate an output file name based on the input file path.

    Parameters:
    input_file_path (str): Path to the input file.

    Returns:
    str: Generated output file name.
    """
    base_name = os.path.basename(input_file_path)
    name, _ = os.path.splitext(base_name)
    parts = name.split('_')
    if len(parts) > 2:
        experiment_number = parts[-1]
        static_actors = parts[-2]
        output_file_name = f"experiment_{experiment_number}_{static_actors}.ply"
    else:
        output_file_name = f"{name}.ply"
    return output_file_name

def save_ply(file_path, points, config):
    """
    Save points to a PLY file.

    Parameters:
    file_path (str): Path to the output PLY file.
    points (numpy.ndarray): Array of point data to save.
    config (Config): Configuration object containing parameters for saving.
    """
    
    # Extract configuration parameters
    float_precision = config.float_precision
    light_color = config.light_color
    material_color = config.material_color
    ply_format = config.ply_format

    if float_precision == 16:
        dtype = np.float16
    elif float_precision == 32:
        dtype = np.float32
    elif float_precision == 64:
        dtype = np.float64
    else:
        raise ValueError("Invalid float precision. Use '16', '32', or '64'.")
    
    header = f"""ply
format {'binary_little_endian' if ply_format == 'Binary' else 'ascii'} 1.0
element vertex {len(points)}
property float{float_precision} x
property float{float_precision} y
property float{float_precision} z
"""
    point_dtype = [('x', dtype), ('y', dtype), ('z', dtype)]
    
    if material_color == "RGBA":
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n"
        point_dtype += [('red', np.uint8), ('green', np.uint8), ('blue', np.uint8), ('alpha', np.uint8)]
    elif material_color == "Grey Scale":
        header += "property uchar intensity\n"
        point_dtype += [('intensity', np.uint8)]
    elif material_color == "RGB":
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        point_dtype += [('red', np.uint8), ('green', np.uint8), ('blue', np.uint8)]
    
    if light_color == "RGBAI":
        header += "property uchar light_red\nproperty uchar light_green\nproperty uchar light_blue\nproperty uchar light_alpha\nproperty uchar light_intensity\n"
        point_dtype += [('light_red', np.uint8), ('light_green', np.uint8), ('light_blue', np.uint8), ('light_alpha', np.uint8), ('light_intensity', np.uint8)]
    elif light_color == "RGBI":
        header += "property uchar light_red\nproperty uchar light_green\nproperty uchar light_blue\nproperty uchar light_intensity\n"
        point_dtype += [('light_red', np.uint8), ('light_green', np.uint8), ('light_blue', np.uint8), ('light_intensity', np.uint8)]
    elif light_color == "RGBA":
        header += "property uchar light_red\nproperty uchar light_green\nproperty uchar light_blue\nproperty uchar light_alpha\n"
        point_dtype += [('light_red', np.uint8), ('light_green', np.uint8), ('light_blue', np.uint8), ('light_alpha', np.uint8)]
    elif light_color == "RGB":
        header += "property uchar light_red\nproperty uchar light_green\nproperty uchar light_blue\n"
        point_dtype += [('light_red', np.uint8), ('light_green', np.uint8), ('light_blue', np.uint8)]
    elif light_color == "Intensity":
        header += "property uchar light_intensity\n"
        point_dtype += [('light_intensity', np.uint8)]
    
    header += "property uchar rendered\n"
    point_dtype += [("rendered", np.uint8)]
    header += "end_header\n"
    
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Ensure points array matches the structured dtype
    structured_points = np.array([tuple(point) for point in points], dtype=point_dtype)
    with open(file_path, 'wb' if ply_format == 'Binary' else 'w') as file:
        file.write(header.encode('utf-8') if ply_format == 'Binary' else header)
        if ply_format == 'Binary':
            structured_points.tofile(file)
        else:
            # Create a format string for ASCII output
            fmt = ' '.join(['%f' if dt[1] == dtype else '%d' for dt in point_dtype])
            np.savetxt(file, structured_points, fmt=fmt)

def check_actor_type(line):
    if line.endswith("(Rectangular Prism):"):
        return "Rectangular Prism"
    elif line.endswith("(Point Cloud):"):
        return "Point Cloud"
    elif line.endswith("(Sphere):"):
        return "Sphere"
    return None

def extract_values(line):
    if line.startswith("Material Color:"):
        values = re.findall(r'\d+', line)
        return "Material Color", [int(value) for value in values]
    elif line.startswith("Light Color:"):
        values = re.findall(r'\d+', line)
        return "Light Color", [int(value) for value in values]
    elif line.startswith("Light Intensity:"):
        return "Light Intensity", float(line.split(":")[1].strip())
    elif line.startswith("Center:"):
        values = line.split()[1:]
        return "Center", [float(value) for value in values]
    elif line.startswith("Radius:"):
        return "Radius", float(line.split(":")[1].strip())
    elif line.startswith("Scale:"):
        values = line.split()[1:]
        return "Scale", [float(value) for value in values]
    elif line.startswith("Rotation:"):
        values = line.split()[1:]
        return "Rotation", [float(value) for value in values]
    elif line.startswith("Closest Light:"):
        return "Closest Light", line.split(":")[1].strip()
    elif line.startswith("Rendered:"):
        value = line.split(":")[1].strip().replace(" ", "")
        if value.lower() == "yes":
            return "Rendered", 1
        elif value.lower() == "no":
            return "Rendered", 0
        return "Rendered", value
    else:
        # Assume it's a point if there are exactly 3 numerical values
        values = line.split()
        if len(values) == 3 and all(re.match(r'-?\d+\.?\d*', v) for v in values):
            return "Points", tuple(float(value) for value in values)
    return None, None

def read_attributes_from_file(file_path, config):
    prisms = []
    spheres = []
    point_clouds = []
    current_actor_type = None
    current_actor_name = None
    current_actor_attributes = {}

    actor_rendering_dict = fetch_visibility_score(file_path, config.dynamic_actors_rendering_dict)

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            actor_type = check_actor_type(line)
            if actor_type:
                # If we already have attributes collected for the previous actor, append it to the appropriate list
                if current_actor_attributes:
                    current_actor_attributes["Name"] = current_actor_name
                    if current_actor_type == "Rectangular Prism":
                        prisms.append(current_actor_attributes)
                    elif current_actor_type == "Sphere":
                        spheres.append(current_actor_attributes)
                    elif current_actor_type == "Point Cloud":
                        point_clouds.append(current_actor_attributes)
                # Reset for the new actor
                current_actor_type = actor_type
                current_actor_name = line.split()[0]
                current_actor_attributes = {"Type": actor_type}
            else:
                line_type, values = extract_values(line)
                if line_type:
                    if line_type == "Points":
                        current_actor_attributes.setdefault("Points", []).append(values)
                    
                    elif line_type =="Rendered" and actor_rendering_dict:
                        current_actor_attributes[line_type] = actor_rendering_dict[current_actor_name]
                    else:
                        current_actor_attributes[line_type] = values

        # Append the last actor's attributes if any
        if current_actor_attributes:
            current_actor_attributes["Name"] = current_actor_name
            if current_actor_type == "Rectangular Prism":
                prisms.append(current_actor_attributes)
            elif current_actor_type == "Sphere":
                spheres.append(current_actor_attributes)
            elif current_actor_type == "Point Cloud":
                point_clouds.append(current_actor_attributes)

    return prisms, spheres, point_clouds
