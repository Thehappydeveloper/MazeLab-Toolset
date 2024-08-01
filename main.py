import os
import sys
import json
import base64
import argparse
from tqdm import tqdm

from main_utils import check_and_create_directory, handle_experiment_participant, Config
from file_operations import generate_dynamic_rendering_dict

def main(config):
    assert os.path.exists(config.dataset_folder_path), f"Dataset folder path '{config.dataset_folder_path}' does not exist."
    check_and_create_directory(config.output_file_path)

    #Generate dict of rendering information for dynamic objects

    for experiment, participants in tqdm(config.selected_experiment_participant_pairs.items(), desc="Experiments"):
        for participant in tqdm(participants, desc=f"Participants in {experiment}", leave=False):
            handle_experiment_participant(experiment, participant, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process point cloud data.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the dataset folder')
    parser.add_argument('--output_path', type=str, required=False, help='Path to the output folder')
    parser.add_argument('--experiments', type=str, required=True, help='Base64 encoded JSON dictionary of experiments and participants')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--material_color', type=str, default="Grey scale", help='Material color')
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
        experiment_dict_json = base64.b64decode(args.experiments).decode('utf-8')
        experiment_dict = json.loads(experiment_dict_json)

        output_path = args.output_path if args.output_path else os.path.join(os.path.dirname(os.path.abspath(__file__)), "pcdDataset")

        dynamic_actors_rendering_dict = generate_dynamic_rendering_dict(args.input_path, experiment_dict)

        config = Config(
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
            normalize=args.normalize_point_cloud,
            dynamic_actors_rendering_dict=dynamic_actors_rendering_dict
        )

        main(config)

    except (ValueError, SyntaxError) as e:
        print(f"Error parsing arguments: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
