def add_visibility_scores(batch_sparse_tensor: SparseConvTensor, batch_sequence_tensor: torch.Tensor, FoV_angle_limits: torch.Tensor, device: str = 'cpu') -> SparseConvTensor:
    """
    Add visibility scores to each point in the sparse tensor for each batch.

    Parameters:
    - batch_sparse_tensor: The input sparse convolution tensor from the spconv library.
    - batch_sequence_tensor: Tensor containing HMD coordinates and angles for each batch.
    - FoV_angle_limits: Angle range of vision (euler angles).
    - device: Device on which to place the resulting tensor.

    Returns:
    - SparseConvTensor with added visibility scores.
    """

    # Extract coordinates and features from sparse tensor
    coordinates = batch_sparse_tensor.indices.to(device)  # Shape: (N, 4), where N is the number of points
    features = batch_sparse_tensor.features.to(device)    # Shape: (N, C), where C is the number of features

    # Precompute the FoV angle limits in radians
    FoV_angle_limits_rad = torch.deg2rad(FoV_angle_limits).to(device)

    # Extract spatial coordinates and batch indices
    spatial_coords = coordinates[:, 1:]
    batch_indices = coordinates[:, 0]

    # Initialize tensor to store visibility scores
    visibility_scores = torch.zeros((features.shape[0], 1), device=device)

    # Compute visibility scores for each point in the batch
    for batch_index in range(batch_sparse_tensor.batch_size):
        # Extract HMD data for the current batch
        HMD_data = batch_sequence_tensor[batch_index].to(device)

        HMD_coordinate = HMD_data[:, :3]
        FoV_angle_set = HMD_data[:, 3:]

        # Apply rotations to get the directional vectors
        direction_vector = apply_rotations(FoV_angle_set, device)

        # Mask to select points belonging to the current batch
        batch_mask = (batch_indices == batch_index)

        # Extract coordinates for points in the current batch
        batch_coordinates = spatial_coords[batch_mask]

        #print(batch_coordinates.shape, HMD_coordinate.unsqueeze(0).shape)

        # Compute vector from HMD to points
        vector_to_points = batch_coordinates - HMD_coordinate

        #print(direction_vector.shape, vector_to_points.shape) # torch.Size([1, 3]) torch.Size([42623, 3])

        # Normalize direction_vector to have the same shape as vector_to_points for broadcasting
        direction_vector = direction_vector.expand_as(vector_to_points)

        # Compute dot product for each component (x, y, z)
        dot_product_x = direction_vector[:, 0] * vector_to_points[:, 0]
        dot_product_y = direction_vector[:, 1] * vector_to_points[:, 1]
        dot_product_z = direction_vector[:, 2] * vector_to_points[:, 2]

        # Compute norms for direction_vector and each vector in vector_to_points
        norm_direction_vector = torch.norm(direction_vector, dim=1)  # shape [42623]
        norm_vector_to_points = torch.norm(vector_to_points, dim=1)   # shape [42623]

        # Compute cosine of angles for each component
        cosine_angle_x = dot_product_x / (norm_direction_vector * norm_vector_to_points)
        cosine_angle_y = dot_product_y / (norm_direction_vector * norm_vector_to_points)
        cosine_angle_z = dot_product_z / (norm_direction_vector * norm_vector_to_points)

        # Clamp cosine values to handle potential numerical issues
        cosine_angle_x = torch.clamp(cosine_angle_x, min=-1.0, max=1.0)
        cosine_angle_y = torch.clamp(cosine_angle_y, min=-1.0, max=1.0)
        cosine_angle_z = torch.clamp(cosine_angle_z, min=-1.0, max=1.0)

        # Compute angles in radians
        angle_x = torch.acos(cosine_angle_x)
        angle_y = torch.acos(cosine_angle_y)
        angle_z = torch.acos(cosine_angle_z)

        # Stack the angles into a single tensor
        angles_between_vectors = torch.stack([angle_x, angle_y, angle_z], dim=1)

        # Check if the angle is within the FoV angle limits
        visibility = (angles_between_vectors < FoV_angle_limits_rad).all(dim=1, keepdim=True).float()

        # Store visibility scores
        visibility_scores[batch_mask] = visibility

    # Concatenate visibility scores with original features
    new_features = torch.cat((features, visibility_scores), dim=1)

    # Create new sparse tensor with updated features
    new_sparse_tensor = SparseConvTensor(new_features, coordinates, batch_sparse_tensor.spatial_shape, batch_sparse_tensor.batch_size)

    #print(f"tensor_a requires_grad: {new_sparse_tensor.requires_grad}")
    #sys.exit()

    return new_sparse_tensor

def apply_rotations(angles: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
    """
    Apply extrinsic rotations to initial forward vectors.

    Parameters:
    - angles: Tensor of shape (batch_size, 3) containing yaw, pitch, and roll angles in degrees for each sample.
    - device: Device on which to place the resulting tensor.

    Returns:
    - 3D direction vectors after applying rotations for each sample in the batch.
    """
    # Convert angles to radians
    angles_rad = torch.deg2rad(angles)

    # Extract individual angles
    yaw = angles_rad[:, 0]
    pitch = angles_rad[:, 1]
    roll = angles_rad[:, 2]

    # Create extrinsic rotation objects for each sample in the batch (same as in UE5)
    direction_vectors = torch.stack([
        torch.cos(pitch) * torch.cos(yaw),
        torch.cos(pitch) * torch.sin(yaw),
        torch.sin(pitch)
    ], dim=1)

    # Set dtype and device
    direction_vectors = direction_vectors.to(dtype=torch.float32, device=device)
    #print(f'direction: {direction_vectors.shape}')

    return direction_vectors

def apply_rotations(angles: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
    """
    Apply extrinsic rotations to initial forward vectors.

    Parameters:
    - angles: Tensor of shape (batch_size, 3) containing yaw, pitch, and roll angles in degrees for each sample.
    - device: Device on which to place the resulting tensor.

    Returns:
    - 3D direction vectors after applying rotations for each sample in the batch.
    """
    # Convert angles to radians
    angles_rad = torch.deg2rad(angles)

    # Extract individual angles
    yaw = angles_rad[:, 0]
    pitch = angles_rad[:, 1]
    roll = angles_rad[:, 2]

    # Create extrinsic rotation objects for each sample in the batch (same as in UE5)
    direction_vectors = torch.stack([
        torch.cos(pitch) * torch.cos(yaw),
        torch.cos(pitch) * torch.sin(yaw),
        torch.sin(pitch)
    ], dim=1)

    # Set dtype and device
    direction_vectors = direction_vectors.to(dtype=torch.float32, device=device)
    #print(f'direction: {direction_vectors.shape}')

    return direction_vectors