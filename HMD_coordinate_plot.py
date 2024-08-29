import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

# Set global font size for all text elements
plt.rcParams.update({'font.size': 10})

# Function to compute the global limits
def compute_global_limits(file_paths):
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df = df[df['Frame'] >= 5].head(2800).reset_index(drop=True)
        position_x = df['Position_X'] / 100
        position_y = df['Position_Y'] / 100
        position_z = df['Position_Z'] / 100

        x_min = min(x_min, position_x.min())
        x_max = max(x_max, position_x.max())
        y_min = min(y_min, position_y.min())
        y_max = max(y_max, position_y.max())
        z_min = min(z_min, position_z.min())
        z_max = max(z_max, position_z.max())

    return x_min, x_max, y_min, y_max, z_min, z_max

# Function to create and save a 3D plot for a given dataset path
def plot_3d_data(file_path, ax, subtitle, x_min, x_max, y_min, y_max, z_min, z_max):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Filter the DataFrame to start from frame 5
    df = df[df['Frame'] >= 5].head(2800).reset_index(drop=True)

    # Extract and scale the relevant columns
    position_x = df['Position_X'] / 100
    position_y = df['Position_Y'] / 100
    position_z = df['Position_Z'] / 100
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

    # Compute shifts to ensure minimum values are zero
    x_shift = -x_min
    y_shift = -y_min
    z_shift = -z_min

    # Apply shifts to position data
    position_x_shifted = position_x + x_shift
    position_y_shifted = position_y + y_shift
    position_z_shifted = position_z + z_shift

    # Plot the line for participant movement
    ax.plot(position_x_shifted, position_y_shifted, position_z_shifted, lw=2, color='blue')

    # Plot direction vectors at intervals with small arrow tips
    interval = 100  # Interval for plotting direction vectors
    arrow_length = 3  # Length of the arrows
    for i in range(0, len(position_x_shifted), interval):
        pos = np.array([position_x_shifted[i], position_y_shifted[i], position_z_shifted[i]])
        dir_vec = direction_vectors[i]
        ax.quiver(pos[0], pos[1], pos[2], dir_vec[0], dir_vec[1], dir_vec[2], color='red', length=arrow_length, normalize=True)

    # Set plot limits
    ax.set_xlim(left=0, right=x_max + x_shift)
    ax.set_ylim(bottom=0, top=y_max + y_shift)
    ax.set_zlim(bottom=0, top=(z_max + z_shift)*3)

    # Set plot labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{subtitle}', y=-0.1)

    # Set the initial viewing angle
    ax.view_init(elev=25, azim=-40)  # Adjust these values to set the initial angle

# Paths to the datasets
file_paths = [
    '/home/jeremy/Documents/datasets/Dataset_large_metadata/Experiments/Experiment_15/DefaultParticipant_20247812121775/HMD_data.csv',
    '/home/jeremy/Documents/datasets/Dataset_large_metadata/Experiments/Experiment_6/DefaultParticipant_202478115514228/HMD_data.csv',
    '/home/jeremy/Documents/datasets/Dataset_large_metadata/Experiments/Experiment_8/DefaultParticipant_20247515313266/HMD_data.csv',
    '/home/jeremy/Documents/datasets/Dataset_large_metadata/Experiments/Experiment_12/DefaultParticipant_202478162545959/HMD_data.csv'
]

# Compute global limits
x_min, x_max, y_min, y_max, z_min, z_max = compute_global_limits(file_paths)

# Create a PDF to save the plots
with PdfPages('/home/jeremy/Documents/datasets/3d_plots.pdf') as pdf:
    fig = plt.figure(figsize=(25, 10))  # Increased figure size
    subtitles = ['(a)', '(b)', '(c)', '(d)']  # Subtitles for the plots
    for i, file_path in enumerate(file_paths):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        plot_3d_data(file_path, ax, subtitles[i], x_min, x_max, y_min, y_max, z_min, z_max)
    
    # Adjust layout to prevent overlap and increase gaps
    plt.subplots_adjust(wspace=0.12)  # Increased space between plots

    # Save the figure as a PDF
    pdf.savefig(fig)
    plt.close()
