"""
SignTalk AI: 93-Node Multimodal Skeletal Graph Topology.

Defines the anatomical landmark graph for Indian Sign Language recognition,
including Left Hand (21 nodes), Right Hand (21 nodes), Upper Pose (11 nodes),
and Facial Contours (40 nodes).

Constructs normalized spatial adjacency matrices supporting:
  - Spatial Configuration Partitioning (K=3 subsets: root/self, centripetal, centrifugal)
  - Distance Partitioning (K=2 subsets: self-loops, 1-hop neighbors)
  - Uniform Partitioning (K=1 subset: symmetrically normalized A + I)
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch


class SignGraph:
    """
    Multimodal Skeletal Graph Topology for 93-node MediaPipe landmark schema.
    
    Attributes:
        num_nodes: Total number of anatomical nodes (V=93).
        edges: List of undirected physical edge tuples (u, v).
        self_loops: List of self-connection tuples (u, u).
        strategy: Adjacency partitioning strategy ('spatial', 'uniform', 'distance').
        A: Precomputed normalized adjacency tensor [K, V, V].
    """

    NUM_NODES = 93

    def __init__(self, strategy: str = "spatial", max_hop: int = 1):
        """
        Initializes the SignGraph.
        
        Args:
            strategy: Partitioning strategy ('spatial', 'uniform', 'distance').
            max_hop: Maximum spatial neighborhood hop distance (default: 1).
        """
        self.num_nodes = self.NUM_NODES
        self.strategy = strategy.lower()
        self.max_hop = max_hop

        self.edges = self._build_edges()
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.root_node = 42  # Nose / cranial anchor as primary kinematic root

        self.hop_dis = self._compute_hop_distance()
        self.A = self._build_adjacency(self.strategy)

    def _build_edges(self) -> List[Tuple[int, int]]:
        """Constructs all 103 undirected anatomical skeletal edges."""
        edges = []

        # -------------------------------------------------------------
        # 1. Left Hand Articulators (Nodes 0 - 20)
        # -------------------------------------------------------------
        # Phalanx chains
        edges.extend([
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
        ])
        # Transverse palm arch
        edges.extend([(5, 9), (9, 13), (13, 17)])

        # -------------------------------------------------------------
        # 2. Right Hand Articulators (Nodes 21 - 41)
        # -------------------------------------------------------------
        # Phalanx chains (offset by +21)
        edges.extend([
            (21, 22), (22, 23), (23, 24), (24, 25), # Thumb
            (21, 26), (26, 27), (27, 28), (28, 29), # Index
            (21, 30), (30, 31), (31, 32), (32, 33), # Middle
            (21, 34), (34, 35), (35, 36), (36, 37), # Ring
            (21, 38), (38, 39), (39, 40), (40, 41)  # Pinky
        ])
        # Transverse palm arch
        edges.extend([(26, 30), (30, 34), (34, 38)])

        # -------------------------------------------------------------
        # 3. Upper Pose Anchors (Nodes 42 - 52)
        # -------------------------------------------------------------
        edges.extend([
            (47, 48),          # Shoulder girdle (Left Shoulder - Right Shoulder)
            (47, 49), (49, 51),# Left Arm: Shoulder -> Elbow -> Wrist Pose
            (48, 50), (50, 52),# Right Arm: Shoulder -> Elbow -> Wrist Pose
            (42, 43), (42, 44),# Nose to Eyes
            (43, 45), (44, 46),# Eyes to Ears
            (42, 47), (42, 48) # Neck/Head to Shoulders
        ])

        # -------------------------------------------------------------
        # 4. Kinematic Hand-to-Body Cross Bridges (2 edges)
        # -------------------------------------------------------------
        edges.extend([
            (0, 51),   # Left Hand Wrist to Left Pose Wrist
            (21, 52)   # Right Hand Wrist to Right Pose Wrist
        ])

        # -------------------------------------------------------------
        # 5. Facial Non-Manual Contours (Nodes 53 - 92)
        # -------------------------------------------------------------
        # Eyebrows (53-60)
        edges.extend([
            (53, 54), (54, 55), (55, 56), # Left brow
            (57, 58), (58, 59), (59, 60), # Right brow
            (53, 43), (56, 42), (57, 42), (60, 44) # Brow to eye/nose anchors
        ])

        # Lips & Mouth Contour (61-76) - 16 node closed cycle
        for i in range(61, 76):
            edges.append((i, i + 1))
        edges.append((76, 61))  # Loop closure
        edges.extend([(61, 42), (69, 42)])  # Mouth to nose anchors

        # Lower Jawline (77-92) - 16 node open chain
        for i in range(77, 92):
            edges.append((i, i + 1))
        edges.extend([(77, 45), (92, 46), (84, 42)])  # Jaw to ear/nose anchors

        # Enforce bidirectional undirected edges
        undirected_edges = set()
        for u, v in edges:
            undirected_edges.add((u, v))
            undirected_edges.add((v, u))

        return sorted(list(undirected_edges))

    def _compute_hop_distance(self) -> np.ndarray:
        """Computes shortest graph distance matrix using Floyd-Warshall."""
        V = self.num_nodes
        hop_dis = np.full((V, V), np.inf, dtype=np.float32)
        np.fill_diagonal(hop_dis, 0)

        for u, v in self.edges:
            hop_dis[u, v] = 1.0
            hop_dis[v, u] = 1.0

        # Floyd-Warshall
        for k in range(V):
            hop_dis = np.minimum(hop_dis, hop_dis[:, k:k+1] + hop_dis[k:k+1, :])

        return hop_dis

    def _build_adjacency(self, strategy: str) -> np.ndarray:
        """
        Constructs partitioned normalized adjacency matrix A.
        
        Returns:
            A: NumPy array of shape [K, V, V] where K is number of partition subsets.
        """
        V = self.num_nodes
        valid_hop = (self.hop_dis <= self.max_hop)

        if strategy == "uniform":
            # Single symmetrically normalized adjacency: D^(-1/2) (A + I) D^(-1/2)
            A = np.zeros((1, V, V), dtype=np.float32)
            A[0] = valid_hop.astype(np.float32)
            D = np.sum(A[0], axis=-1)
            D_inv_sqrt = np.zeros_like(D)
            mask = (D > 0)
            D_inv_sqrt[mask] = np.power(D[mask], -0.5)
            A[0] = D_inv_sqrt[:, None] * A[0] * D_inv_sqrt[None, :]
            return A

        elif strategy == "distance":
            # 2 subsets: 0 = self-loops, 1 = 1-hop neighbors
            A = np.zeros((2, V, V), dtype=np.float32)
            for i in range(V):
                for j in range(V):
                    if self.hop_dis[i, j] == 0:
                        A[0, i, j] = 1.0
                    elif self.hop_dis[i, j] == 1:
                        A[1, i, j] = 1.0
            # Row degree normalization: D^(-1) A
            for k in range(2):
                D = np.sum(A[k], axis=-1)
                D_inv = np.zeros_like(D)
                mask = (D > 0)
                D_inv[mask] = 1.0 / D[mask]
                A[k] = D_inv[:, None] * A[k]
            return A

        elif strategy == "spatial":
            # 3 subsets (Spatial Configuration Partitioning):
            # Subset 0: Root/Self-connection
            # Subset 1: Centripetal (neighbors closer to root node than current node)
            # Subset 2: Centrifugal (neighbors further from or equal to root node)
            A = np.zeros((3, V, V), dtype=np.float32)
            root = self.root_node
            dist_to_root = self.hop_dis[root]

            for i in range(V):
                for j in range(V):
                    if not valid_hop[i, j]:
                        continue
                    if self.hop_dis[i, j] == 0:
                        A[0, i, j] = 1.0
                    elif dist_to_root[j] < dist_to_root[i]:
                        A[1, i, j] = 1.0
                    else:
                        A[2, i, j] = 1.0

            # Row-normalize each partition subset: D^(-1) A_k
            for k in range(3):
                D = np.sum(A[k], axis=-1)
                D_inv = np.zeros_like(D)
                mask = (D > 0)
                D_inv[mask] = 1.0 / D[mask]
                A[k] = D_inv[:, None] * A[k]
            return A

        else:
            raise ValueError(f"Unknown graph adjacency strategy: '{strategy}'. Choose 'spatial', 'distance', or 'uniform'.")

    def get_adjacency_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Returns the precomputed adjacency matrix as a PyTorch FloatTensor [K, V, V]."""
        tensor_A = torch.from_numpy(self.A).float()
        if device is not None:
            tensor_A = tensor_A.to(device)
        return tensor_A

    def validate_input_nodes(self, num_input_nodes: int) -> None:
        """Validates that input tensor node dimension matches graph node count."""
        if num_input_nodes != self.num_nodes:
            raise ValueError(
                f"[SignGraph Error] Node dimension mismatch: Model received input with {num_input_nodes} nodes, "
                f"but graph topology requires exactly {self.num_nodes} nodes."
            )
