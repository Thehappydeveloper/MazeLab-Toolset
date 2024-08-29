import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = '/home/jeremy/Documents/datasets/Dataset_large_metadata/Experiments/Experiment_15/DefaultParticipant_20247812121775/HMD_data.csv'
df = pd.read_csv(file_path)

# Filter the DataFrame to start from frame 5 and limit to 2800 frames
df = df[df['Frame'] >= 5].head(2800).reset_index(drop=True)

# Extract the relevant columns
position_x = df['Position_X']
position_y = df['Position_Y']

# Create the 2D plot
fig, ax = plt.subplots()

# Plot the line for participant movement
ax.plot(position_x, position_y, lw=2, color='blue')

# Set plot limits
ax.set_xlim(position_x.min(), position_x.max())
ax.set_ylim(position_y.min(), position_y.max())

# Set plot labels
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('2D Line Plot of Participant Movements')

plt.show()
