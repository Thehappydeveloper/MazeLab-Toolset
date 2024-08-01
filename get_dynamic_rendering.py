import os
import re
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

def process_experiment_dataset(base_path):
    """
    Processes the dataset to extract and transform rendering states for all actors across all frames,
    given a base path.

    Args:
        base_path (str): Path to the base dataset directory.

    Returns:
        dict: A nested dictionary with the structure {Experiment: {Participant: {Actor: [rendering state counts]}}}.
    """
    data = {}
    
    # Assuming the structure is base_path/Experiments
    experiments_path = os.path.join(base_path, 'Experiments')
    experiments = [exp for exp in os.listdir(experiments_path) if os.path.isdir(os.path.join(experiments_path, exp))]

    for experiment in tqdm(experiments, desc="Processing Experiments"):
        experiment_path = os.path.join(experiments_path, experiment)
        
        participants_data = {}
        participants = [part for part in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, part))]

        for participant in tqdm(participants, desc=f"Processing Participants in {experiment}", leave=False):
            participant_path = os.path.join(experiment_path, participant)

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

# Path to the base dataset directory
base_path = "/home/jeremy/Documents/datasets/Dataset_large_metadata/"
result = process_experiment_dataset(base_path)

# Print or use the result as needed
print(result)
