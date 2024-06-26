import os
import numpy as np
import re
import sys

def load_point_cloud(file_path):
    """
    Load point cloud data from a file.

    Parameters:
    file_path (str): Path to the point cloud file.
    num_attributes (int): Number of attributes to read from the file.

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

def save_ply(file_path, points, float_precision, light_color, material_color, ply_format='Binary'):
    """
    Save points to a PLY file.

    Parameters:
    file_path (str): Path to the output PLY file.
    points (numpy.ndarray): Array of point data to save.
    float_precision (str): String representing the floating precision ("16", "32", or "64").
    light_color (str): The type of light color data ("RGBI", "RGB", "Intensity").
    material_color (str): The type of material color data ("RGBA", "Grey Scale", "RGB").
    format (str): The format of the PLY file, either 'binary' or 'ascii'.
    """
    
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
    elif material_color == "Grey scale":
        header += "property uchar intensity\n"
        point_dtype += [('intensity', np.uint8)]
    elif material_color == "RGB":
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        point_dtype += [('red', np.uint8), ('green', np.uint8), ('blue', np.uint8)]
    
    if light_color == "RGBI":
        header += "property uchar light_red\nproperty uchar light_green\nproperty uchar light_blue\nproperty uchar light_intensity\n"
        point_dtype += [('light_red', np.uint8), ('light_green', np.uint8), ('light_blue', np.uint8), ('light_intensity', np.uint8)]
    elif light_color == "RGB":
        header += "property uchar light_red\nproperty uchar light_green\nproperty uchar light_blue\n"
        point_dtype += [('light_red', np.uint8), ('light_green', np.uint8), ('light_blue', np.uint8)]
    elif light_color == "Intensity":
        header += "property uchar light_intensity\n"
        point_dtype += [('light_intensity', np.uint8)]
    
    header += "property uchar rendered\n"
    point_dtype +=  [("rendered", np.uint8)]
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

def read_attributes_from_file(file_path):
    prisms = []
    spheres = []
    point_clouds = []
    current_actor_type = None
    current_actor_name = None
    current_actor_attributes = {}

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
