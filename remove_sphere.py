import os

def remove_lines(file_path):
    """Remove 'SplinePath (Sphere):' line and the next 7 lines from the file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    with open(file_path, 'w') as file:
        i = 0
        while i < len(lines):
            if lines[i].strip() == "SplinePath (Sphere):":
                i += 8  # Skip this line and the next 7 lines
            else:
                file.write(lines[i])
                i += 1

def process_directory(root_dir):
    """Process all relevant files in the directory."""
    experiments_dir = os.path.join(root_dir, 'Experiments')
    for experiment in os.listdir(experiments_dir):
        print(experiment)
        if experiment == 'Experiment_6':
            continue
        experiment_dir = os.path.join(experiments_dir, experiment)
        if os.path.isdir(experiment_dir):
            for participant in os.listdir(experiment_dir):
                participant_dir = os.path.join(experiment_dir, participant)
                dynamic_actors_dir = os.path.join(participant_dir, 'DynamicActors')
                if os.path.isdir(dynamic_actors_dir):
                    for filename in os.listdir(dynamic_actors_dir):
                        if filename.startswith("frame_") and filename.endswith(".txt"):
                            file_path = os.path.join(dynamic_actors_dir, filename)
                            remove_lines(file_path)

# Example usage
root_directory = '/home/jeremy/Documents/BACKUP/collected_fov_data/Dataset_large'
process_directory(root_directory)
