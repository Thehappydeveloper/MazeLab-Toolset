import os
from scipy.spatial.transform import Rotation as R
import sys
import argparse
import traceback
import json
import base64
import time

from file_operations import save_ply
from frame_processing import process_frame, process_frame_by_actor

from tqdm import tqdm  # tqdm for progress bars

def main(dataset_folder_path, output_file_path, selected_experiment_participant_pairs,
         sphere_density=0.1, prism_density=0.02, 
         include_spheres="Yes", include_prisms="Yes", 
         include_point_clouds="Yes", FPS=30, material_color="Grey scale", 
         light_color="Grey scale", float_precision=32,
         ply_format="Binary", pcds_point_cap=100000, normalize="No"):
    
    try:
        # Check if dataset folder path exists
        if not os.path.exists(dataset_folder_path):
            raise FileNotFoundError(f"Dataset folder path '{dataset_folder_path}' does not exist.")

        # Create output directory if it doesn't exist
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)

        # Get point cloud directory from dataset folder path
        point_cloud_directory = os.path.join(dataset_folder_path, 'PCDs', 'Static')

        # Progress bar for the main loop
        experiments_progress = tqdm(selected_experiment_participant_pairs.items(), desc="Experiments", file=sys.stdout)
        
        for experiment, participants in experiments_progress:
            participants_progress = tqdm(participants, desc=f"Participants in {experiment}", file=sys.stdout, leave=False)
            
            for participant in participants_progress:
                try:
                    # Define the paths
                    HMD_csv_path = os.path.join(dataset_folder_path, 'Experiments', experiment, participant, 'HMD_data.csv')
                    viewport_json_path = os.path.join(dataset_folder_path, 'Experiments', experiment, participant, 'staticActorsFoV.json')

                    # Destination folder
                    json_csv_destination = os.path.join(output_file_path, experiment, participant)

                    # Ensure the destination directory exists
                    os.makedirs(json_csv_destination, exist_ok=True)

                    # Copy the files
                    with open(HMD_csv_path, 'rb') as src_file:
                        with open(os.path.join(json_csv_destination, 'HMD_data.csv'), 'wb') as dest_file:
                            dest_file.write(src_file.read())

                    with open(viewport_json_path, 'rb') as src_file:
                        with open(os.path.join(json_csv_destination, 'staticActorsFoV.json'), 'wb') as dest_file:
                            dest_file.write(src_file.read())

                    frames_path = os.path.join(dataset_folder_path, 'Experiments', experiment, participant, 'DynamicActors')

                    if not os.path.exists(frames_path):
                        raise FileNotFoundError(f"Frames path '{frames_path}' does not exist for experiment '{experiment}' and participant '{participant}'.")

                    # Get static point cloud from dataset folder path
                    static_actors_path = os.path.join(dataset_folder_path, 'Metadata', 'StaticActors', experiment, 'StaticActors.txt')

                    # Log static actors processing start time
                    start_time = time.time()
                    tqdm.write(f"Processing static actors for experiment '{experiment}', participant '{participant}'...")
                    
                    static_points_by_actor = process_frame_by_actor(static_actors_path, material_color, light_color, include_spheres, include_prisms, include_point_clouds,
                                                                    sphere_density, prism_density, point_cloud_directory, pcds_point_cap, normalize)
                    
                    # Log static actors processing end time and duration
                    end_time = time.time()
                    processing_time = end_time - start_time
                    tqdm.write(f"Static actors processed in {processing_time:.2f} seconds.")
                    
                    # Progress bar for saving static actor PLY files
                    static_progress = tqdm(static_points_by_actor.items(), desc="Saving Static PCDs", file=sys.stdout, leave=False)
                    
                    for name, points in static_progress:
                        output_path = os.path.join(output_file_path, experiment, "StaticPCDs", (name + ".ply"))
                        try:
                            save_ply(output_path, points, float_precision, light_color, material_color, ply_format)
                            static_progress.set_postfix(file=name)
                        except Exception as e:
                            tqdm.write(f"Error saving file '{output_path}': {e}", file=sys.stderr)
                            tqdm.write(traceback.format_exc(), file=sys.stderr)

                    # Progress bar for frames processing
                    frames_progress = tqdm(sorted(os.listdir(frames_path)), desc=f"Frames in {participant}", file=sys.stdout, leave=False)
                    
                    # Loop through the frames in the folder
                    for frame_name in frames_progress:
                        frame_path = os.path.join(frames_path, frame_name)
                        # Calculate the frame number from the frame name
                        frame_number = int((frame_name.split('.')[0]).split('_')[1])
                        
                        # Skip frames based on the FPS value
                        if FPS in [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60] and frame_number % (60 // FPS) != 0:
                            continue  # Skip frames that are not multiples of (60 // FPS)
                        
                        frame_points = process_frame(frame_path, material_color, light_color, include_spheres, include_prisms,
                                                    include_point_clouds, sphere_density, prism_density, point_cloud_directory, pcds_point_cap, normalize)

                        output_path = os.path.join(output_file_path, experiment, participant, "DynamicActors", (frame_name.split('.')[0] + ".ply"))
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        save_ply(output_path, frame_points, float_precision, light_color, material_color, ply_format)
                        frames_progress.set_postfix(frame=frame_name)  # Update the frame name in the progress bar

                except FileNotFoundError as e:
                    tqdm.write(f"Error in processing experiment '{experiment}', participant '{participant}': {e}", file=sys.stderr)
                    tqdm.write(traceback.format_exc(), file=sys.stderr)
                except ValueError as e:
                    tqdm.write(f"Value error in processing experiment '{experiment}', participant '{participant}': {e}", file=sys.stderr)
                    tqdm.write(traceback.format_exc(), file=sys.stderr)
                except Exception as e:
                    tqdm.write(f"Unexpected error in processing experiment '{experiment}', participant '{participant}': {e}", file=sys.stderr)
                    tqdm.write(traceback.format_exc(), file=sys.stderr)

    except FileNotFoundError as e:
        tqdm.write(f"Error: {e}", file=sys.stderr)
        tqdm.write(traceback.format_exc(), file=sys.stderr)
    except ValueError as e:
        tqdm.write(f"Error: {e}", file=sys.stderr)
        tqdm.write(traceback.format_exc(), file=sys.stderr)
    except Exception as e:
        tqdm.write(f"Unexpected error: {e}", file=sys.stderr)
        tqdm.write(traceback.format_exc(), file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process point cloud data.')

    parser.add_argument('--input_path', type=str, required=True, help='Path to the dataset folder')
    parser.add_argument('--output_path', type=str, required=False, help='Path to the output folder')
    parser.add_argument('--experiments', type=str, required=True, help='Dictionary of experiments and participants')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--material_color', type=str, default="Grey scale", help='Material Color')
    parser.add_argument('--light_color', type=str, default="Grey scale", help='Light color')
    parser.add_argument('--float_precision', type=int, default=32, help='Float precision')
    parser.add_argument('--sphere_density', type=float, default=0.1, help='Sphere density')
    parser.add_argument('--prism_density', type=float, default=0.02, help='Prism density')
    parser.add_argument('--sphere_inclusion', type=str, default="Yes", help='Include spheres')
    parser.add_argument('--prisms_inclusion', type=str, default="Yes", help='Include prisms')
    parser.add_argument('--point_clouds_inclusion', type=str, default="Yes", help='Include point clouds')
    parser.add_argument('--pcds_point_cap', type=int, default=100000, help='Total number of points for PCDs')
    parser.add_argument('--ply_format', type=str, default="Binary", help='PLY as binary or ASCII')
    parser.add_argument('--normalize_point_cloud', type=str, default="No", help='Normalize the produced PLYs')

    args = parser.parse_args()

    try:
        # Decode the base64 string and then parse the JSON
        experiment_dict_base64 = args.experiments
        experiment_dict_json = base64.b64decode(experiment_dict_base64).decode('utf-8')
        experiment_dict = json.loads(experiment_dict_json)

        # Default output path if not specified
        if not args.output_path:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(script_dir, "pcdDataset")
        else:
            output_path = args.output_path

        main(
            dataset_folder_path=args.input_path,
            output_file_path=output_path,
            selected_experiment_participant_pairs=experiment_dict,
            sphere_density=args.sphere_density,
            prism_density=args.prism_density,
            include_spheres=args.sphere_inclusion,
            include_prisms=args.prisms_inclusion,
            include_point_clouds=args.point_clouds_inclusion,
            FPS=args.fps,
            material_color=args.material_color,
            light_color=args.light_color,
            float_precision=args.float_precision,
            ply_format=args.ply_format,
            pcds_point_cap=args.pcds_point_cap,
            normalize=args.normalize_point_cloud
        )

    except (ValueError, SyntaxError) as e:
        print(f"Error parsing arguments: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)