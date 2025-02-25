import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# Load the CSV file into a DataFrame
file_path = '/home/jeremy/Documents/datasets/Dataset_large_metadata/Experiments/Experiment_10/DefaultParticipant_202478111612890/HMD_data.csv'
df = pd.read_csv(file_path)

# Filter the DataFrame to start from frame 5
df = df[df['Frame'] >= 5].reset_index(drop=True)

# Extract the relevant columns
frame = df['Frame']
position_x = df['Position_X']
position_y = df['Position_Y']
position_z = df['Position_Z']
quat_x = df['Quat_X']
quat_y = df['Quat_Y']
quat_z = df['Quat_Z']
quat_w = df['Quat_W']

# Convert quaternions to direction vectors
def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w
    yy = y * y
    yz = y * z
    yw = y * w
    zz = z * z
    zw = z * w

    R = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
        [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
        [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)]
    ])
    return R

def apply_quaternions(quaternions):
    direction_vectors = np.zeros((quaternions.shape[0], 3))
    forward_vector = np.array([1, 0, 0])  # Forward vector along x-axis

    for i in range(quaternions.shape[0]):
        R = quaternion_to_rotation_matrix(quaternions[i])
        direction_vectors[i] = R @ forward_vector

    return direction_vectors

quaternions = np.column_stack((quat_x, quat_y, quat_z, quat_w))
direction_vectors = apply_quaternions(quaternions)

# Function to create the grid of tiles
def create_tiles(min_xyz, max_xyz, tile_shape):
    x_range = np.linspace(min_xyz[0], max_xyz[0], tile_shape[0] + 1)
    y_range = np.linspace(min_xyz[1], max_xyz[1], tile_shape[1] + 1)
    z_range = np.linspace(min_xyz[2], max_xyz[2], tile_shape[2] + 1)
    tiles = []

    for i in range(tile_shape[0]):
        for j in range(tile_shape[1]):
            for k in range(tile_shape[2]):
                x_min = x_range[i]
                x_max = x_range[i + 1]
                y_min = y_range[j]
                y_max = y_range[j + 1]
                z_min = z_range[k]
                z_max = z_range[k + 1]
                tiles.append((x_min, x_max, y_min, y_max, z_min, z_max))
    
    return tiles

# Function to determine if a tile is visible based on direction vector and viewport limits
def is_tile_visible(tile, position, direction_vector, fov_horizontal, fov_vertical):
    tile_center = np.array([
        (tile[0] + tile[1]) / 2,
        (tile[2] + tile[3]) / 2,
        (tile[4] + tile[5]) / 2
    ])
    
    # Compute direction vector
    direction = direction_vector / np.linalg.norm(direction_vector)
    to_tile = tile_center - position
    to_tile_dist = np.linalg.norm(to_tile)
    to_tile = to_tile / to_tile_dist  # Normalize to get direction
    
    # Compute dot product and angle
    dot_product = np.dot(direction, to_tile)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # Convert angles to degrees
    angle_degrees = np.degrees(angle)
    
    # Check if angle is within the field of view and apply distance weighting
    visible = angle_degrees <= (fov_horizontal / 2) and angle_degrees <= (fov_vertical / 2)
    if visible:
        # Use inverse distance to weight the visibility
        distance_weight = 1 / (to_tile_dist + 1e-6)  # Add small value to avoid division by zero
        return distance_weight
    return 0

# Function to determine the color of a tile based on direction vector
def get_tile_color(tile, direction_vector, position):
    tile_center = np.array([
        (tile[0] + tile[1]) / 2,
        (tile[2] + tile[3]) / 2,
        (tile[4] + tile[5]) / 2
    ])
    direction = direction_vector / np.linalg.norm(direction_vector)
    to_tile = tile_center - position
    to_tile_dist = np.linalg.norm(to_tile)
    to_tile = to_tile / to_tile_dist
    
    dot_product = np.dot(direction, to_tile)
    color_intensity = 1 - (dot_product + 1) / 2  # Normalize to range [0, 1]
    
    # Apply distance-based weight to color
    distance_weight = 1 / (to_tile_dist + 1e-6)
    weighted_intensity = color_intensity * distance_weight

    # Ensure intensity is between 0 and 1
    weighted_intensity = np.clip(weighted_intensity, 0, 1)
    
    # Map visibility weight to a color gradient from blue to red
    color = plt.cm.coolwarm(weighted_intensity)
    return color

# Create the grid of tiles
tile_shape = [4, 4, 1]  # Reduced resolution
min_xyz = [position_x.min(), position_y.min(), position_z.min()]
max_xyz = [position_x.max(), position_y.max(), position_z.max()]
tiles = create_tiles(min_xyz, max_xyz, tile_shape)

# Define the field of view (horizontal and vertical)
fov_horizontal = 97  # degrees
fov_vertical = 93  # degrees

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the plot line and quiver list
line, = ax.plot([], [], [], lw=2)
quiver_objects = []

# Create scatter plot for tiles
tile_scatter = ax.scatter([], [], [], c=[], s=100, edgecolors='w')

# Set plot limits
ax.set_xlim((position_x.min(), position_x.max()))
ax.set_ylim((position_y.min(), position_y.max()))
ax.set_zlim((position_z.min(), position_y.max()-position_y.min()))

# Set plot labels
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Line Plot of Participant Movements')

# Initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    # Remove all quiver objects from the plot
    for obj in quiver_objects:
        obj.remove()
    quiver_objects.clear()
    # Initialize tiles only once
    tile_scatter._offsets3d = ([], [], [])
    return line, tile_scatter

# Animation function: called sequentially
def animate(i):
    # Update the line with positions
    line.set_data(position_x[:i], position_y[:i])
    line.set_3d_properties(position_z[:i])

    # Update the direction vector
    if i > 0:
        current_pos = np.array([position_x[i], position_y[i], position_z[i]])
        current_dir = direction_vectors[i]

        # Remove old quivers
        for obj in quiver_objects:
            obj.remove()
        quiver_objects.clear()

        # Add new quiver
        quiver = ax.quiver(current_pos[0], current_pos[1], current_pos[2],
                           current_dir[0], current_dir[1], current_dir[2],
                           color='red', length=1000, normalize=False)
        quiver_objects.append(quiver)

        # Update tiles color
        tile_offsets = [[], [], []]
        tile_colors = []

        for tile in tiles:
            visibility_weight = is_tile_visible(tile, current_pos, current_dir, fov_horizontal, fov_vertical)
            if visibility_weight > 0:
                color = get_tile_color(tile, current_dir, current_pos)
                tile_colors.append(color)
                # Compute the center of the tile for placement
                tile_center = np.array([
                    (tile[0] + tile[1]) / 2,
                    (tile[2] + tile[3]) / 2,
                    (tile[4] + tile[5]) / 2
                ])
                tile_offsets[0].append(tile_center[0])
                tile_offsets[1].append(tile_center[1])
                tile_offsets[2].append(tile_center[2])

        # Update tile scatter plot
        tile_scatter._offsets3d = tuple(tile_offsets)
        tile_scatter.set_edgecolor(tile_colors)
        
    return line, tile_scatter

# Wrapper function to show progress
def tqdm_animate():
    with tqdm(total=len(frame), desc='Animating Frames') as pbar:
        ani = animation.FuncAnimation(fig, animate, init_func=init,
                                      frames=len(frame), interval=50, blit=False)
        ani.save('participant_movements_with_direction_and_tiles.mp4', writer='ffmpeg')
        for i in range(len(frame)):
            pbar.update(1)
            animate(i)

# Call the wrapper function
tqdm_animate()

plt.show()
