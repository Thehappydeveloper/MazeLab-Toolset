import torch
from spconv.pytorch import SparseConvTensor

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

    R = torch.tensor([
        [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
        [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
        [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)]
    ], dtype=torch.float32)
    return R

def apply_quaternions(quaternion):
    forward_vector = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)  # Forward vector along x-axis
    R = quaternion_to_rotation_matrix(quaternion)
    direction_vector = torch.matmul(R, forward_vector)
    return direction_vector

def is_points_visible(points, position, direction_vector, fov_horizontal, fov_vertical):
    """
    Determine visibility of multiple points based on a single direction vector and viewport limits.

    Parameters:
    - points: Tensor of shape (N, 3) where N is the number of points.
    - position: Tensor of shape (3,) representing the HMD position.
    - direction_vector: Tensor of shape (3,) representing the direction vector.
    - fov_horizontal: Horizontal field of view in radians.
    - fov_vertical: Vertical field of view in radians.

    Returns:
    - Visibility scores tensor of shape (N,) where each entry is the visibility score for the corresponding point.
    """
    # Normalize direction vector
    direction_vector = direction_vector / direction_vector.norm()

    # Compute vectors from position to points
    to_points = points - position
    to_points_dist = to_points.norm(dim=1)
    to_points = to_points / (to_points_dist.unsqueeze(1) + 1e-6)  # Normalize directions, avoid division by zero

    # Compute dot products
    dot_products = torch.matmul(to_points, direction_vector)

    # Compute angles
    angles = torch.acos(torch.clamp(dot_products, -1.0, 1.0))
    angle_degrees = torch.degrees(angles)

    # Check if angles are within the field of view
    visible_horizontal = angle_degrees <= (torch.degrees(fov_horizontal) / 2)
    visible_vertical = angle_degrees <= (torch.degrees(fov_vertical) / 2)

    # Determine if points are visible based on both FOV constraints
    visibility = visible_horizontal & visible_vertical

    # Use inverse distance to weight the visibility
    distance_weights = 1 / (to_points_dist + 1e-6)

    # Compute final visibility scores
    visibility_scores = visibility.float() * distance_weights

    return visibility_scores

def add_visibility_scores(batch_sparse_tensor: SparseConvTensor, batch_sequence_tensor: torch.Tensor, FoV_angle_limits: torch.Tensor, device: str = 'cpu') -> SparseConvTensor:
    """
    Add visibility scores to each point in the sparse tensor for each batch.

    Parameters:
    - batch_sparse_tensor: The input sparse convolution tensor from the spconv library.
    - batch_sequence_tensor: Tensor containing HMD coordinates and angles for each batch.
    - FoV_angle_limits: Angle range of vision (Euler angles).
    - device: Device on which to place the resulting tensor.

    Returns:
    - SparseConvTensor with added visibility scores.
    """
    # Extract coordinates and features from sparse tensor (already on device)
    coordinates = batch_sparse_tensor.indices  # Shape: (N, 4), where N is the number of points
    features = batch_sparse_tensor.features    # Shape: (N, C), where C is the number of features

    # Precompute the FoV angle limits in radians
    FoV_angle_limits_rad = torch.deg2rad(FoV_angle_limits).to(device)

    # Extract spatial coordinates and batch indices
    spatial_coords = coordinates[:, 1:].float()  # Ensure spatial coordinates are floats
    batch_indices = coordinates[:, 0].int()      # Ensure batch indices are ints

    # Initialize tensor to store visibility scores
    visibility_scores = torch.zeros((features.shape[0], 1), device=device)

    # Compute visibility scores for each point in the batch
    for batch_index in range(batch_sparse_tensor.batch_size):
        # Extract HMD data for the current batch
        HMD_data = batch_sequence_tensor[batch_index]
        HMD_coordinate = HMD_data[:3]
        quaternions = HMD_data[3:]

        # Apply rotations to get the directional vector
        direction_vector = apply_quaternions(quaternions).to(device)

        # Mask to select points belonging to the current batch
        batch_mask = (batch_indices == batch_index)

        # Extract coordinates for points in the current batch
        batch_coordinates = spatial_coords[batch_mask]

        # Compute visibility scores for the batch
        visibility_scores_batch = is_points_visible(batch_coordinates, HMD_coordinate, direction_vector, FoV_angle_limits_rad[0], FoV_angle_limits_rad[1])

        # Store visibility scores
        visibility_scores[batch_mask] = visibility_scores_batch.unsqueeze(1)

    # Concatenate visibility scores with original features
    new_features = torch.cat((features, visibility_scores), dim=1)

    # Create new sparse tensor with updated features
    new_sparse_tensor = SparseConvTensor(new_features, coordinates, batch_sparse_tensor.spatial_shape, batch_sparse_tensor.batch_size)

    return new_sparse_tensor
