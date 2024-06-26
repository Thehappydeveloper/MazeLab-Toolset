import numpy as np
from scipy.spatial.transform import Rotation as R

def apply_transformations(points, center, scale, rotation, point_cap):
    """
    Apply scaling, rotation, and translation transformations to points.

    Parameters:
    points (numpy.ndarray): Array of (x, y, z, ...) coordinates.
    center (tuple): (x, y, z) coordinates of the translation center.
    scale (tuple): (sx, sy, sz) scaling factors.
    rotation (tuple): (rx, ry, rz) rotation angles in degrees.

    Returns:
    numpy.ndarray: Array of transformed points.
    """
    nb_points = len(points)
    if(point_cap < nb_points): 
        np.random.seed(42)
        np.random.shuffle(points)
        points = points[:int(point_cap)]
    #inverted_rotation = [-angle for angle in rotation]
    rotation = (rotation[0],rotation[1],rotation[2]+180)
    # Negate the X-axis
    points[:, 0] = -points[:, 0]
    quat = R.from_euler('xyz', rotation, degrees=True).as_quat()
    points_rotated = R.from_quat(quat).apply(points[:, :3])  # Apply rotation only to first three dimensions
    points_scaled = points_rotated * np.array(scale)
    points_translated = points_scaled + np.array(center)
    return np.concatenate((points_translated, points[:, 3:]), axis=1)  # Concatenate with unchanged dimensions