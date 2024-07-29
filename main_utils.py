import os
import sys
import time
import traceback
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from file_operations import save_ply
from frame_processing import process_frame, process_frame_by_actor

class Config:
    def __init__(self, dataset_folder_path, output_file_path, selected_experiment_participant_pairs,
                 sphere_density=0.1, prism_density=0.02, include_spheres="Yes", include_prisms="Yes", 
                 include_point_clouds="Yes", FPS=30, material_color="Grey scale", light_color="Grey scale", 
                 float_precision=32, ply_format="Binary", pcds_point_cap=100000, normalize="No"):
        self.dataset_folder_path = dataset_folder_path
        self.output_file_path = output_file_path
        self.selected_experiment_participant_pairs = selected_experiment_participant_pairs
        self.sphere_density = sphere_density
        self.prism_density = prism_density
        self.include_spheres = include_spheres
        self.include_prisms = include_prisms
        self.include_point_clouds = include_point_clouds
        self.FPS = FPS
        self.material_color = material_color
        self.light_color = light_color
        self.float_precision = float_precision
        self.ply_format = ply_format
        self.pcds_point_cap = pcds_point_cap
        self.normalize = normalize

def check_and_create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def copy_file(src, dest):
    with open(src, 'rb') as src_file:
        with open(dest, 'wb') as dest_file:
            dest_file.write(src_file.read())

def handle_experiment_participant(experiment, participant, config):
    try:
        experiment_path = os.path.join(config.dataset_folder_path, 'Experiments', experiment, participant)
        HMD_csv_path = os.path.join(experiment_path, 'HMD_data.csv')
        viewport_json_path = os.path.join(experiment_path, 'staticActorsFoV.json')
        json_csv_destination = os.path.join(config.output_file_path, experiment, participant)

        check_and_create_directory(json_csv_destination)
        copy_file(HMD_csv_path, os.path.join(json_csv_destination, 'HMD_data.csv'))
        copy_file(viewport_json_path, os.path.join(json_csv_destination, 'staticActorsFoV.json'))

        frames_path = os.path.join(experiment_path, 'DynamicActors')
        assert os.path.exists(frames_path), f"Frames path '{frames_path}' does not exist."

        static_actors_path = os.path.join(config.dataset_folder_path, 'Metadata', 'StaticActors', experiment, 'StaticActors.txt')
        static_output_path = os.path.join(config.output_file_path, experiment, "StaticPCDs")
        check_and_create_directory(static_output_path)

        process_static_actors(static_actors_path, static_output_path, config)

        dynamic_output_path = os.path.join(config.output_file_path, experiment, participant, "DynamicActors")
        check_and_create_directory(dynamic_output_path)

        process_frames(frames_path, dynamic_output_path, config)

    except AssertionError as e:
        tqdm.write(f"Error in processing experiment '{experiment}', participant '{participant}': {e}", file=sys.stderr)
        tqdm.write(traceback.format_exc(), file=sys.stderr)
    except Exception as e:
        tqdm.write(f"Unexpected error in processing experiment '{experiment}', participant '{participant}': {e}", file=sys.stderr)
        tqdm.write(traceback.format_exc(), file=sys.stderr)

def process_static_actors(static_actors_path, output_path, config):
    start_time = time.time()
    tqdm.write(f"Processing static actors...")

    static_points_by_actor = process_frame_by_actor(static_actors_path, config)
    
    end_time = time.time()
    tqdm.write(f"Static actors processed in {end_time - start_time:.2f} seconds.")
    
    for name, points in tqdm(static_points_by_actor.items(), desc="Saving Static PCDs", leave=False):
        output_ply_path = os.path.join(output_path, f"{name}.ply")
        try:
            save_ply(output_ply_path, points, config)
        except Exception as e:
            tqdm.write(f"Error saving file '{output_ply_path}': {e}", file=sys.stderr)
            tqdm.write(traceback.format_exc(), file=sys.stderr)

def process_frames(frames_path, output_path, config):
    for frame_name in tqdm(sorted(os.listdir(frames_path)), desc="Processing frames", leave=False):
        frame_path = os.path.join(frames_path, frame_name)
        frame_number = int((frame_name.split('.')[0]).split('_')[1])
        
        if config.FPS in [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60] and frame_number % (60 // config.FPS) != 0:
            continue
        
        frame_points = process_frame(frame_path, config)

        output_ply_path = os.path.join(output_path, f"{frame_name.split('.')[0]}.ply")
        check_and_create_directory(os.path.dirname(output_ply_path))
        save_ply(output_ply_path, frame_points, config)
