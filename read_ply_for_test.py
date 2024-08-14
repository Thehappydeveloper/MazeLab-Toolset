import pandas as pd
from plyfile import PlyData

def read_ply_to_dataframe(file_path):
    plydata = PlyData.read(file_path)
    
    # Extract data from the ply file
    vertex_data = plydata['vertex'].data

    # Create a DataFrame from the ply data
    df = pd.DataFrame(vertex_data)
    
    return df

# Example usage
file_path = '/home/jeremy/Documents/datasets/pcdDataset/Experiment_3/DefaultParticipant_2024711173727614/DynamicActors/frame_4254.ply'  # Replace with your PLY file path
df = read_ply_to_dataframe(file_path)

# Get all unique values in the 'rendered' column
unique_rendered_values = df['rendered'].unique()

print("Unique values in the 'rendered' column:")
print(unique_rendered_values)