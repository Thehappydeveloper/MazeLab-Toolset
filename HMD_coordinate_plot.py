import pandas as pd
import plotly.graph_objs as go
import numpy as np

# Load the CSV file into a DataFrame
file_path = '/home/jeremy/Documents/BACKUP/collected_fov_data/Dataset_large/Experiments/Experiment_10/DefaultParticipant_20247812252941/HMD_data.csv'  # Update with the actual path to your CSV file
df = pd.read_csv(file_path)

# Filter the DataFrame to start from frame 5
df = df[df['Frame'] >= 5].reset_index(drop=True)

# Extract the relevant columns
frame = df['Frame']
position_x = df['Position_X']
position_y = df['Position_Y']
position_z = df['Position_Z']

# Normalize the frame numbers to [0, 1] for color mapping
frame_normalized = (frame - frame.min()) / (frame.max() - frame.min())

# Create a colormap that transitions from blue to red
colors = [f'rgb({int(r*255)}, 0, {int(b*255)})' for r, b in zip(frame_normalized, 1-frame_normalized)]

# Create the 3D line plot with Plotly
fig = go.Figure()

for i in range(len(frame) - 1):
    fig.add_trace(go.Scatter3d(
        x=position_x.iloc[i:i+2],
        y=position_y.iloc[i:i+2],
        z=position_z.iloc[i:i+2],
        mode='lines',
        line=dict(color=colors[i], width=2),
        showlegend=False  # Hide the legend for each trace
    ))

# Ensure the axes are scaled equally
max_range = np.array([position_x.max()-position_x.min(), position_y.max()-position_y.min(), position_z.max()-position_z.min()]).max() / 2.0
mean_x = position_x.mean()
mean_y = position_y.mean()
mean_z = position_z.mean()

fig.update_layout(title='3D Line Plot with Color Gradient')

fig.show()
