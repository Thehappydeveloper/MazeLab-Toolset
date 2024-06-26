import numpy as np
import os
import sys
from file_operations import read_attributes_from_file, load_point_cloud
from sphere_converter import generate_sphere_points
from rect_prism_converter import generate_prism_faces
from pcd_converter import apply_transformations

def process_points(points, color, light, material_color, light_color, rendered, all_points, actor_name=None, all_points_by_actor=None):
    """
    Processes a set of points, applying color and light transformations based on the specified material and light color modes.
    Optionally, organizes points by actor if actor_name and all_points_by_actor are provided.

    Args:
        points (np.ndarray or list): List or array of points to process.
        color (np.ndarray or list): Color values for the points.
        light (np.ndarray or list): Light values for the points.
        material_color (str): Mode for material color processing ('RGBA', 'Grey Scale', 'RGB').
        light_color (str): Mode for light color processing ('RGBA', 'Grey Scale', 'RGB').
        rendered (str): Whether the points should be rendered ('yes' or 'no').
        all_points (list): List to store processed points.
        actor_name (str, optional): Name of the actor for organizing points. Default is None.
        all_points_by_actor (dict, optional): Dictionary to store points organized by actor. Default is None.
    """
    transformed_points = [] 

    is_color_list = isinstance(color, np.ndarray) and len(color) == len(points)
    is_light_list = isinstance(light, np.ndarray) and len(light) == len(points)

    for i, point in enumerate(points):
        point = np.append(point, []) if isinstance(point, np.ndarray) else point[:]
        point_color = color[i] if is_color_list else color
        point_light = light[i] if is_light_list else light

        # Process material color
        if material_color == "RGBA":
            point = np.append(point, point_color)
        elif material_color == "Grey Scale":
            Y = 0.299 * point_color[0] + 0.587 * point_color[1] + 0.114 * point_color[2]
            point = np.append(point, Y)
        elif material_color == "RGB":
            point = np.append(point, point_color[:3])

        # Process light color
        if light_color == "RGBI":
            point = np.append(point, point_light)
        elif light_color == "Grey Scale":
            Y = 0.299 * point_light[0] + 0.587 * point_light[1] + 0.114 * point_light[2]
            point = np.append(point, Y)
        elif light_color == "RGB":
            point = np.append(point, point_light[:3])

        if rendered == "no":
            point = np.append(point, 0)
        else:
            point = np.append(point, 1)

        transformed_points.append(point)

    if actor_name is not None and all_points_by_actor is not None:
        if actor_name not in all_points_by_actor:
            all_points_by_actor[actor_name] = []
        all_points_by_actor[actor_name].extend(transformed_points)
    else:
        all_points.extend(transformed_points)

def process_entities(entities, entity_type, density, material_color, light_color, all_points, actor_processing=False, all_points_by_actor=None):
    """
    Processes entities (spheres, prisms, or point clouds) by generating their points and applying the necessary transformations.

    Args:
        entities (list): List of entities to process.
        entity_type (str): Type of the entity ('sphere', 'prism', 'point_cloud').
        density (int): Density parameter for point generation.
        material_color (str): Mode for material color processing ('RGBA', 'Grey Scale', 'RGB').
        light_color (str): Mode for light color processing ('RGBA', 'Grey Scale', 'RGB').
        all_points (list): List to store processed points.
        actor_processing (bool, optional): Flag to indicate if points should be organized by actor. Default is False.
        all_points_by_actor (dict, optional): Dictionary to store points organized by actor. Default is None.
    """
    if entity_type == "point_cloud":
        nb_pcds = len(entities)
        if nb_pcds > 0:
            point_budget_pcds = int(density)
            points_per_pcd = int(point_budget_pcds / nb_pcds)

    for entity in entities:
        rendered = entity["Rendered"]
        actor_name = entity["Name"] if actor_processing else None
        if entity_type == 'sphere':
            points = generate_sphere_points(entity["Center"], entity["Radius"], density, True)
            process_points(points, entity["Material Color"], entity["Light Color"], material_color, light_color, rendered, all_points, actor_name, all_points_by_actor)
        elif entity_type == 'prism':
            points = generate_prism_faces(entity["Points"], density)
            process_points(points, entity["Material Color"], entity["Light Color"], material_color, light_color, rendered, all_points, actor_name, all_points_by_actor)
        elif entity_type == 'point_cloud':
            cloud_file_path = os.path.join(entity["Directory"], f'{entity["Name"]}.txt')
            if os.path.exists(cloud_file_path):
                cloud_points = load_point_cloud(cloud_file_path)
                if len(cloud_points) > points_per_pcd:
                    np.random.seed(42)
                    np.random.shuffle(cloud_points)
                    cloud_points = cloud_points[:points_per_pcd]
                points = apply_transformations(cloud_points, entity['Center'], entity['Scale'], entity['Rotation'], density / len(entities))
                process_points(points[:, :3], points[:, 3:7], entity["Light Color"], material_color, light_color, rendered, all_points, actor_name, all_points_by_actor)
            else:
                print(f"Warning: Point cloud file {cloud_file_path} not found.")
                continue

def normalize_points(points):
    """
    Normalizes the given points by centering them and scaling to fit within a unit sphere.

    Args:
        points (np.ndarray): Array of points to normalize.

    Returns:
        np.ndarray: Normalized points.
    """
    if points.size == 0:
        print("No points to save.")
        return points

    centroid = np.mean(points[:, :3], axis=0)
    points[:, :3] -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points[:, :3]) ** 2, axis=-1)))
    points[:, :3] /= furthest_distance

    return points

def process_frame(frame_path, material_color, light_color, include_spheres, include_prisms, include_point_clouds, sphere_density, prism_density, point_cloud_directory, pcds_point_cap, normalize):
    """
    Processes a frame by reading entities (spheres, prisms, point clouds) from the frame file, transforming their points,
    and optionally normalizing the points.

    Args:
        frame_path (str): Path to the frame file.
        material_color (str): Mode for material color processing ('RGBA', 'Grey Scale', 'RGB').
        light_color (str): Mode for light color processing ('RGBA', 'Grey Scale', 'RGB').
        include_spheres (str): Whether to include spheres ('Yes' or 'No').
        include_prisms (str): Whether to include prisms ('Yes' or 'No').
        include_point_clouds (str): Whether to include point clouds ('Yes' or 'No').
        sphere_density (int): Density for sphere point generation.
        prism_density (int): Density for prism point generation.
        point_cloud_directory (str): Directory containing point cloud files.
        pcds_point_cap (int): Maximum number of points for point clouds.
        normalize (str): Whether to normalize points ('Yes' or 'No').

    Returns:
        np.ndarray: Processed and optionally normalized points.
    """
    prisms, spheres, point_clouds = read_attributes_from_file(frame_path)
    all_points = []

    if include_spheres == 'Yes':
        process_entities(spheres, 'sphere', sphere_density, material_color, light_color, all_points)

    if include_prisms == 'Yes':
        process_entities(prisms, 'prism', prism_density, material_color, light_color, all_points)

    if include_point_clouds == 'Yes':
        for pcd in point_clouds:
            pcd["Directory"] = point_cloud_directory

        process_entities(point_clouds, 'point_cloud', pcds_point_cap, material_color, light_color, all_points)

    all_points = np.array(all_points, dtype=np.float32)

    if normalize == 'Yes':
        all_points = normalize_points(all_points)

    return all_points

def process_frame_by_actor(frame_path, material_color, light_color, include_spheres, include_prisms, include_point_clouds, sphere_density, prism_density, point_cloud_directory, pcds_point_cap, normalize):
    """
    Processes a frame by reading entities (spheres, prisms, point clouds) from the frame file, transforming their points,
    organizing points by actor, and optionally normalizing the points.

    Args:
        frame_path (str): Path to the frame file.
        material_color (str): Mode for material color processing ('RGBA', 'Grey Scale', 'RGB').
        light_color (str): Mode for light color processing ('RGBA', 'Grey Scale', 'RGB').
        include_spheres (str): Whether to include spheres ('Yes' or 'No').
        include_prisms (str): Whether to include prisms ('Yes' or 'No').
        include_point_clouds (str): Whether to include point clouds ('Yes' or 'No').
        sphere_density (int): Density for sphere point generation.
        prism_density (int): Density for prism point generation.
        point_cloud_directory (str): Directory containing point cloud files.
        pcds_point_cap (int): Maximum number of points for point clouds.
        normalize (str): Whether to normalize points ('Yes' or 'No').

    Returns:
        dict: Dictionary with actor names as keys and their corresponding processed and optionally normalized points as values.
    """
    prisms, spheres, point_clouds = read_attributes_from_file(frame_path)
    all_points_by_actor = {}

    if include_spheres == 'Yes':
        process_entities(spheres, 'sphere', sphere_density, material_color, light_color, None, True, all_points_by_actor)

    if include_prisms == 'Yes':
        process_entities(prisms, 'prism', prism_density, material_color, light_color, None, True, all_points_by_actor)

    if include_point_clouds == 'Yes':
        for pcd in point_clouds:
            pcd["Directory"] = point_cloud_directory
        process_entities(point_clouds, 'point_cloud', pcds_point_cap, material_color, light_color, None, True, all_points_by_actor)

    for actor_name in all_points_by_actor:
        all_points_by_actor[actor_name] = np.array(all_points_by_actor[actor_name], dtype=np.float32)
        if normalize == 'Yes' and all_points_by_actor[actor_name].size > 0:
            all_points_by_actor[actor_name] = normalize_points(all_points_by_actor[actor_name])

    return all_points_by_actor
