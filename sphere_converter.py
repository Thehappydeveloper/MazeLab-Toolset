import numpy as np

def fibonacci_sphere(samples, randomize=True):
    """
    Generate points on a sphere using the Fibonacci lattice method.

    Parameters:
    samples (int): Number of points to generate.
    randomize (bool): Whether to randomize the distribution of points.

    Returns:
    list of tuple: List of (x, y, z) coordinates of points on the sphere.
    """
    points = []
    offset = 2.0 / samples
    increment = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y**2)
        phi = ((i + (1 if randomize else 0)) % samples) * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append((x, y, z))

    return points

def scale_points(points, center, radius):
    """
    Scale and translate points to fit a sphere of a given radius and center.

    Parameters:
    points (list of tuple): List of (x, y, z) coordinates of points.
    center (tuple): (x, y, z) coordinates of the sphere center.
    radius (float): Radius of the sphere.

    Returns:
    numpy.ndarray: Array of scaled and translated points.
    """
    scaled_points = []
    for point in points:
        x = center[0] + radius * point[0]
        y = center[1] + radius * point[1]
        z = center[2] + radius * point[2]
        scaled_points.append((x, y, z))
    return np.array(scaled_points)

def calculate_number_of_points(density, radius):
    """
    Calculate the number of points required for a sphere based on density.

    Parameters:
    density (float): Density of points per unit area.
    radius (float): Radius of the sphere.

    Returns:
    int: Number of points required.
    """
    surface_area = 4 * np.pi * radius**2
    return int(density * np.sqrt(surface_area))

def generate_sphere_points(center, radius, density, randomize=True):
    """
    Generate points on a sphere's surface.

    Parameters:
    center (tuple): (x, y, z) coordinates of the sphere center.
    radius (float): Radius of the sphere.
    density (float): Density of points per unit area.
    randomize (bool): Whether to randomize the distribution of points.

    Returns:
    numpy.ndarray: Array of points on the sphere.
    """
    num_points = calculate_number_of_points(density, radius)
    surface_points = fibonacci_sphere(num_points, randomize=randomize)
    return scale_points(surface_points, center, radius)