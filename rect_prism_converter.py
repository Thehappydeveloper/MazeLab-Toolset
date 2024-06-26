import numpy as np
import sys

def order_points_3d(points):
    """
    Order points in a consistent manner for generating faces of a prism.

    Parameters:
    points (list of tuple): List of (x, y, z) coordinates of points.

    Returns:
    numpy.ndarray: Array of ordered points.
    """
    points = np.array(points)
    sorted_points = points[np.argsort(points[:, 2])]
    bottom_face = sorted(sorted_points[:4], key=lambda p: (p[0], p[1]))
    top_face = sorted(sorted_points[4:], key=lambda p: (p[0], p[1]))
    return np.vstack((bottom_face, top_face))

def order_points_rect(points):
    """
    Order points counterclockwise starting from the bottom-left.
    
    Args:
    - points (numpy.ndarray): Array of 3D points defining the rectangle.

    Returns:
    - numpy.ndarray: Array of points ordered counterclockwise starting from the bottom-left.
    """
    
    def calculate_centroid(points):
        """
        Calculate the centroid of a set of points.
        
        Args:
        - points (numpy.ndarray): Array of points.
        
        Returns:
        - numpy.ndarray: Centroid coordinates.
        """
        return np.mean(points, axis=0)

    def compute_normal(p1, p2, p3):
        """
        Compute the normal vector of a plane defined by three points.
        
        Args:
        - p1, p2, p3 (numpy.ndarray): Points defining the plane.
        
        Returns:
        - numpy.ndarray: Normal vector of the plane.
        """
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        return normal / np.linalg.norm(normal)

    def project_points(points, origin, normal):
        """
        Project points onto a plane defined by its origin and normal vector.
        
        Args:
        - points (numpy.ndarray): Array of points to project.
        - origin (numpy.ndarray): Origin point of the plane.
        - normal (numpy.ndarray): Normal vector of the plane.
        
        Returns:
        - numpy.ndarray: Projected points in 2D (x, y) coordinates.
        """
        z_axis = normal
        x_axis = np.cross(np.array([0, 0, 1]), z_axis)
        if np.linalg.norm(x_axis) == 0:
            x_axis = np.cross(np.array([0, 1, 0]), z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        transform = np.array([x_axis, y_axis, z_axis])
        
        translated_points = points - origin
        projected_points = translated_points @ transform.T
        
        return projected_points[:, :2]

    def convert_to_polar(points):
        """
        Convert 2D Cartesian coordinates to polar coordinates relative to the centroid.
        
        Args:
        - points (numpy.ndarray): Array of 2D points.
        
        Returns:
        - numpy.ndarray: Angles in radians corresponding to the points.
        """
        angles = np.arctan2(points[:, 1], points[:, 0])
        return angles

    centroid = calculate_centroid(points)
    normal = compute_normal(points[0], points[1], points[2])
    projected_points = project_points(points, centroid, normal)
    angles = convert_to_polar(projected_points)
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    
    return sorted_points

def interpolate_face(p1, p2, p3, p4, points_per_unit_length):
    """
    Interpolate points on a plane defined by a rectangle in 3D space.
    
    Args:
    - p1, p2, p3, p4 (numpy.ndarray): Coordinates of the four corners of the rectangle.
    - points_per_unit_length (float): Number of points per unit length for interpolation.

    Returns:
    - numpy.ndarray: Interpolated points on the rectangle's plane.
    """
    
    def interpolate(p1, p2, p3, p4, u, v):
        """
        Interpolate points on the plane defined by the rectangle.
        
        Args:
        - p1, p2, p3, p4 (numpy.ndarray): Coordinates of the four corners of the rectangle.
        - u (numpy.ndarray): Parameter array for interpolation along the first edge.
        - v (numpy.ndarray): Parameter array for interpolation along the second edge.
        
        Returns:
        - numpy.ndarray: Interpolated points on the plane.
        """
        p1 = p1[:, np.newaxis, np.newaxis]
        p2 = p2[:, np.newaxis, np.newaxis]
        p3 = p3[:, np.newaxis, np.newaxis]
        p4 = p4[:, np.newaxis, np.newaxis]
        return (1 - u) * (1 - v) * p1 + u * (1 - v) * p2 + u * v * p3 + (1 - u) * v * p4

    points = np.array([p1, p2, p3, p4])
  
    # Order the points correctly
    ordered_points = order_points_rect(points)

    # Calculate edge lengths
    edge1 = np.linalg.norm(ordered_points[1] - ordered_points[0])
    edge2 = np.linalg.norm(ordered_points[2] - ordered_points[1])
    
    # Calculate number of points based on edge lengths and points_per_unit_length
    num_points_u = int(points_per_unit_length * edge1) + 1
    num_points_v = int(points_per_unit_length * edge2) + 1
    
    # Create a parameter grid in the (u, v) space of the rectangle
    u = np.linspace(0, 1, num_points_u)
    v = np.linspace(0, 1, num_points_v)

    uu, vv = np.meshgrid(u, v)

    # Generate points on the plane
    grid_points = interpolate(ordered_points[0], ordered_points[1], ordered_points[2], ordered_points[3], uu, vv)

    return np.vstack((grid_points[0].flatten(), grid_points[1].flatten(), grid_points[2].flatten())).T

def generate_prism_faces(points, points_per_unit_length):
    """
    Generate interpolated points on the faces of a rectangular prism.

    Parameters:
    points (list of tuple): List of (x, y, z) coordinates of the prism corners.
    points_per_unit_length (int): Number of points per unit length of the edges.

    Returns:
    numpy.ndarray: Array of interpolated points on the prism faces.
    """
    ordered_points = order_points_3d(points)

    faces = [
        [ordered_points[0], ordered_points[1], ordered_points[3], ordered_points[2]], # Bottom face
        [ordered_points[4], ordered_points[5], ordered_points[7], ordered_points[6]], # Top face
        [ordered_points[0], ordered_points[1], ordered_points[5], ordered_points[4]], # side
        [ordered_points[2], ordered_points[3], ordered_points[7], ordered_points[6]], # side
        [ordered_points[3], ordered_points[1], ordered_points[5], ordered_points[7]], # side
        [ordered_points[0], ordered_points[4], ordered_points[6], ordered_points[2]]  # side
    ]

    interpolated_faces = []
    for face in faces:
        interpolated_points = interpolate_face(face[0], face[1], face[2], face[3], points_per_unit_length)
        interpolated_faces.append(interpolated_points)

    return np.vstack(interpolated_faces)
