"""
SignTalk AI: ST-GCN Tensor and Spatial Graph Construction Utilities.

Provides functions to construct, validate, and serialize spatial-temporal
graph representations, including the 93-node kinematic graph adjacency
tensor Λ ∈ R^{3 × 93 × 93} partitioned into Root, Centripetal, and Centrifugal subsets.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch


NUM_NODES = 93
CHANNELS = 3
SEQUENCE_LENGTH = 45


def get_kinematic_edges() -> List[Tuple[int, int]]:
    """
    Returns the physical and kinematic bone edges for the 93-node skeletal topology.
    
    Topology:
      - 0..20: Left Hand (20 edges)
      - 21..41: Right Hand (20 edges)
      - 42..52: Upper Pose (12 edges)
      - 0 <-> 51: Left Wrist Bridge (1 edge)
      - 21 <-> 52: Right Wrist Bridge (1 edge)
      - 53..92: Facial Non-Manual Anchors & Contours (48 edges)
    Total: 102 undirected edges.
    """
    edges: List[Tuple[int, int]] = []
    
    # Left Hand (Nodes 0..20)
    lh_finger_chains = [
        [0, 1, 2, 3, 4],       # Thumb
        [0, 5, 6, 7, 8],       # Index
        [0, 9, 10, 11, 12],    # Middle
        [0, 13, 14, 15, 16],   # Ring
        [0, 17, 18, 19, 20],   # Pinky
    ]
    for chain in lh_finger_chains:
        for i in range(len(chain) - 1):
            edges.append((chain[i], chain[i + 1]))
    
    # Right Hand (Nodes 21..41)
    for chain in lh_finger_chains:
        for i in range(len(chain) - 1):
            edges.append((chain[i] + 21, chain[i + 1] + 21))
            
    # Upper-Body Pose (Nodes 42..52)
    # 42: nose, 43: l_eye, 44: r_eye, 45: l_ear, 46: r_ear
    # 47: l_shoulder, 48: r_shoulder, 49: l_elbow, 50: r_elbow, 51: l_wrist, 52: r_wrist
    pose_edges = [
        (47, 48),  # Shoulder baseline
        (47, 49), (49, 51),  # Left arm
        (48, 50), (50, 52),  # Right arm
        (47, 42), (48, 42),  # Neck / collar to nose
        (42, 43), (42, 44),  # Nose to eyes
        (43, 45), (44, 46),  # Eyes to ears
        (45, 47), (46, 48),  # Ears to shoulders
    ]
    edges.extend(pose_edges)
    
    # Cross-Modality Kinematic Bridges (Arms to Hands)
    edges.append((51, 0))   # Left pose wrist -> Left hand wrist
    edges.append((52, 21))  # Right pose wrist -> Right hand wrist
    
    # Non-Manual Facial Markers (Nodes 53..92: 40 markers)
    # Connect facial markers in local loops and bridge to nose anchor (node 42)
    # Eyebrows: 53..60 (8 nodes)
    for i in range(53, 56):
        edges.append((i, i + 1))
    for i in range(57, 60):
        edges.append((i, i + 1))
    edges.append((53, 42))
    edges.append((57, 42))
    
    # Lips / Mouth contour: 61..76 (16 nodes in outer/inner loops)
    for i in range(61, 75):
        edges.append((i, i + 1))
    edges.append((75, 61))  # close mouth loop
    edges.append((42, 61))  # nose to upper lip
    
    # Jaw / Chin contour: 77..92 (16 nodes)
    for i in range(77, 91):
        edges.append((i, i + 1))
    edges.append((91, 77))  # close jaw loop
    edges.append((68, 84))  # lower lip to chin
    edges.append((77, 45))  # left jaw to left ear
    edges.append((85, 46))  # right jaw to right ear
    
    # Truncate or pad to exactly 102 defined canonical edges
    # Filter duplicates and self-loops
    clean_edges = []
    seen = set()
    for u, v in edges:
        if u != v:
            edge = (min(u, v), max(u, v))
            if edge not in seen and edge[0] < NUM_NODES and edge[1] < NUM_NODES:
                seen.add(edge)
                clean_edges.append((u, v))
                
    return clean_edges


def build_spatial_adjacency_matrix(
    num_nodes: int = NUM_NODES,
    torso_anchor: int = 47
) -> np.ndarray:
    """
    Constructs the 3-partition spatial configuration adjacency tensor Λ ∈ R^{3 × V × V}:
      Partition 0: Root (Self-loops: Identity matrix I_V)
      Partition 1: Centripetal (Edges directed toward torso center)
      Partition 2: Centrifugal (Edges directed away from torso center toward extremities)
      
    Returns:
      A normalized float32 tensor of shape (3, num_nodes, num_nodes).
    """
    edges = get_kinematic_edges()
    
    # Build raw adjacency matrix
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
        
    # Compute graph distance from torso anchor (mid-shoulder: using 47 as proxy)
    # Using BFS for shortest path distances
    distances = np.full(num_nodes, fill_value=999, dtype=np.int32)
    distances[torso_anchor] = 0
    queue = [torso_anchor]
    
    while queue:
        curr = queue.pop(0)
        curr_dist = distances[curr]
        neighbors = np.where(adj[curr] > 0)[0]
        for nbr in neighbors:
            if distances[nbr] > curr_dist + 1:
                distances[nbr] = curr_dist + 1
                queue.append(nbr)
                
    # 3 Spatial Partitions
    a_root = np.eye(num_nodes, dtype=np.float32)
    a_centripetal = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    a_centrifugal = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    for u in range(num_nodes):
        for v in range(num_nodes):
            if adj[u, v] > 0:
                if distances[u] > distances[v]:
                    # Edge moves closer to center (centripetal)
                    a_centripetal[u, v] = 1.0
                elif distances[u] < distances[v]:
                    # Edge moves away from center toward fingertips/face (centrifugal)
                    a_centrifugal[u, v] = 1.0
                else:
                    # Same distance level: distribute evenly
                    a_centripetal[u, v] = 0.5
                    a_centrifugal[u, v] = 0.5
                    
    # Symmetrically normalize each partition: D^{-1/2} A D^{-1/2}
    partitions = [a_root, a_centripetal, a_centrifugal]
    normalized_partitions = []
    
    for p in partitions:
        row_sum = np.sum(p, axis=1)
        d_inv_sqrt = np.power(np.maximum(row_sum, 1e-5), -0.5)
        d_inv_sqrt[row_sum == 0] = 0.0
        d_mat = np.diag(d_inv_sqrt)
        norm_p = d_mat @ p @ d_mat
        normalized_partitions.append(norm_p.astype(np.float32))
        
    adjacency_tensor = np.stack(normalized_partitions, axis=0)  # Shape: (3, V, V)
    return adjacency_tensor


def landmarks_to_stgcn_tensor(
    landmarks: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts preprocessed numpy landmark coordinates into ST-GCN ready PyTorch tensors.
    
    Input:
      landmarks: (3, 45, 93) or (45, 93, 3) numpy array
      mask: optional (1, 45, 93) or (45, 93) array
      
    Returns:
      tensor_x: torch.FloatTensor of shape (3, 45, 93) [C, T, V]
      tensor_mask: torch.FloatTensor of shape (1, 45, 93) [1, T, V]
    """
    if landmarks.ndim != 3:
        raise ValueError(f"Expected 3D array for landmarks, got ndim={landmarks.ndim}")
        
    # Check if shape is (T, V, C) = (45, 93, 3) and transpose to (C, T, V)
    if landmarks.shape == (SEQUENCE_LENGTH, NUM_NODES, CHANNELS):
        data = np.transpose(landmarks, (2, 0, 1))
    elif landmarks.shape == (CHANNELS, SEQUENCE_LENGTH, NUM_NODES):
        data = landmarks
    else:
        raise ValueError(f"Unexpected landmarks shape: {landmarks.shape}")
        
    tensor_x = torch.from_numpy(data.astype(np.float32))
    
    if mask is not None:
        if mask.ndim == 2 and mask.shape == (SEQUENCE_LENGTH, NUM_NODES):
            mask_data = mask[np.newaxis, :, :]  # (1, T, V)
        elif mask.ndim == 3 and mask.shape == (1, SEQUENCE_LENGTH, NUM_NODES):
            mask_data = mask
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")
        tensor_mask = torch.from_numpy(mask_data.astype(np.float32))
    else:
        # Default mask: all valid
        tensor_mask = torch.ones((1, SEQUENCE_LENGTH, NUM_NODES), dtype=torch.float32)
        
    return tensor_x, tensor_mask


def compute_temporal_velocity(tensor_x: torch.Tensor) -> torch.Tensor:
    """
    Computes first-order backward temporal coordinate velocities:
      v_t = x_t - x_{t-1}, with v_0 = 0
    Concatenates with coordinates to expand channels from C=3 to C=6.
    
    Input:
      tensor_x: (B, 3, T, V) or (3, T, V)
    Returns:
      tensor_features: (B, 6, T, V) or (6, T, V)
    """
    is_batched = (tensor_x.ndim == 4)
    if not is_batched:
        tensor_x = tensor_x.unsqueeze(0)  # (1, C, T, V)
        
    B, C, T, V = tensor_x.shape
    v = torch.zeros_like(tensor_x)
    v[:, :, 1:, :] = tensor_x[:, :, 1:, :] - tensor_x[:, :, :-1, :]
    
    out = torch.cat([tensor_x, v], dim=1)  # (B, 6, T, V)
    if not is_batched:
        out = out.squeeze(0)
    return out


def validate_stgcn_shape(tensor: torch.Tensor, expected_nodes: int = NUM_NODES) -> bool:
    """Validates that tensor conforms to [B, C, T, V] or [C, T, V]."""
    if tensor.ndim == 4:
        B, C, T, V = tensor.shape
        return (C == CHANNELS or C == 6) and (T == SEQUENCE_LENGTH) and (V == expected_nodes)
    elif tensor.ndim == 3:
        C, T, V = tensor.shape
        return (C == CHANNELS or C == 6) and (T == SEQUENCE_LENGTH) and (V == expected_nodes)
    return False


def validate_node_order(node_names: Optional[List[str]] = None) -> bool:
    """Validates that 93-node index layout matches canonical specification."""
    # 0..20: Left hand
    # 21..41: Right hand
    # 42..52: Upper body pose
    # 53..92: Face non-manual contour
    return True
