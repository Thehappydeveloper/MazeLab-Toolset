import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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
def plot_3d_data(file_path, ax, x_min, x_max, y_min, y_max, z_min, z_max):
    df = pd.read_csv(file_path)
    df = df[df['Frame'] >= 5].head(2800).reset_index(drop=True)

    position_x = df['Position_X'] / 100
    position_y = df['Position_Y'] / 100
    position_z = df['Position_Z'] / 100
    quat_x = df['Quat_X']
    quat_y = df['Quat_Y']
    quat_z = df['Quat_Z']
    quat_w = df['Quat_W']

    def quaternion_to_rotation_matrix(q):
        x, y, z, w = q
        R = np.array([
            [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
            [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
            [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)]
        ])
        return R

    def apply_quaternions(quaternions):
        direction_vectors = np.zeros((quaternions.shape[0], 3))
        forward_vector = np.array([1, 0, 0])

        for i in range(quaternions.shape[0]):
            R = quaternion_to_rotation_matrix(quaternions[i])
            direction_vectors[i] = R @ forward_vector

        return direction_vectors

    quaternions = np.column_stack((quat_x, quat_y, quat_z, quat_w))
    direction_vectors = apply_quaternions(quaternions)

    x_shift = -x_min
    y_shift = -y_min
    z_shift = -z_min

    position_x_shifted = position_x + x_shift
    position_y_shifted = position_y + y_shift
    position_z_shifted = position_z + z_shift

    ax.plot(position_x_shifted, position_y_shifted, position_z_shifted, lw=2, color='#4285F4')

    interval = 100
    arrow_length = 3
    for i in range(0, len(position_x_shifted), interval):
        pos = np.array([position_x_shifted[i], position_y_shifted[i], position_z_shifted[i]])
        dir_vec = direction_vectors[i]
        ax.quiver(pos[0], pos[1], pos[2], dir_vec[0], dir_vec[1], dir_vec[2], color='#0F9D58', length=arrow_length, normalize=True)

    ax.set_xlim(left=0, right=x_max + x_shift)
    ax.set_ylim(bottom=0, top=y_max + y_shift)
    ax.set_zlim(bottom=0, top=(z_max + z_shift)*3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=25, azim=-40)

# Paths to the datasets
file_paths = [
    '/home/jeremy/Documents/datasets/Dataset_large_metadata/Experiments/Experiment_15/DefaultParticipant_20247812121775/HMD_data.csv'
]

# Compute global limits
x_min, x_max, y_min, y_max, z_min, z_max = compute_global_limits(file_paths)

# Create a PDF to save the plot
with PdfPages('/home/jeremy/Documents/datasets/3d_plot_single.pdf') as pdf:
    fig = plt.figure(figsize=(8, 8))  # Adjusted figure size for a single plot
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot_3d_data(file_paths[0], ax, x_min, x_max, y_min, y_max, z_min, z_max)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
