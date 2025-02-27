import torch
import numpy as np
from scipy.sparse.csgraph import dijkstra
import math

# Negative sampling
def negative_sampling(matrix, balanced_atom=False):

    matrix = torch.tensor(matrix, dtype=torch.float32)
    num_vertices, num_edges = matrix.shape
    incidence_matrix_neg = torch.zeros_like(matrix)

    for e in range(num_edges):
        edge = matrix[:, e]  

        nonzero_nodes = torch.where(edge != 0)[0]
        nonzero_values = edge[nonzero_nodes]

        if len(nonzero_nodes) <= 1:
            continue

        num_remove = math.floor(len(nonzero_nodes) * 0.5)

        perm = torch.randperm(len(nonzero_nodes))
        removed_nodes = nonzero_nodes[perm[:num_remove]]
        kept_nodes = nonzero_nodes[perm[num_remove:]]
        
        removed_values = edge[removed_nodes]

        nodes_comp = torch.where(edge == 0)[0]
        
        if balanced_atom:
            used_nodes = torch.where(torch.any(matrix != 0, dim=1))[0]
            nodes_comp = torch.tensor([n for n in nodes_comp if n not in used_nodes])

        if len(nodes_comp) < len(removed_nodes):
            continue

        perm_comp = torch.randperm(len(nodes_comp))
        new_nodes = nodes_comp[perm_comp[:len(removed_nodes)]]

        incidence_matrix_neg[kept_nodes, e] = edge[kept_nodes]
        incidence_matrix_neg[new_nodes, e] = removed_values

    combined_matrix = torch.cat((matrix, incidence_matrix_neg), dim=1)
    return combined_matrix, incidence_matrix_neg.shape[1]

# Compute s-adjacency matrix
def edge_adjacency_matrix(incidence_matrix: torch.Tensor, s: int) -> torch.Tensor:
    if not isinstance(incidence_matrix, torch.Tensor):
        incidence_matrix = torch.tensor(incidence_matrix, dtype=torch.float32)
    M = torch.abs(incidence_matrix)

    product = M.T @ M
    product.fill_diagonal_(1)
    adj_matrix = (product >= s).int()
    return adj_matrix

def shortest_node_distances(incidence_matrix, s):
    """
    Compute the shortest distance between nodes in a hypergraph.

    Parameters:
    - incidence_matrix (numpy.ndarray): A #node * #edge incidence matrix.
    - s.

    Returns:
    - torch.Tensor: A #node * #node matrix where element (i, j) represents the shortest distance between node i and node j.
    """
    # Step 1: Compute edge-to-edge shortest distances using Dijkstra algorithm
    e_adjacency_matrix = edge_adjacency_matrix(incidence_matrix, s)
    # print(f"e_adjaceny_matrix;{e_adjacency_matrix}")
    e_adjacency_np = e_adjacency_matrix.cpu().numpy()

    # Use scipy.sparse.csgraph.dijkstra to compute shortest distances
    shortest_distances = dijkstra(e_adjacency_np, directed=False)

    # Convert the result back to PyTorch tensor
    edge_distance_matrix = torch.from_numpy(shortest_distances).float()

    # Handle infinite distances (optional normalization)
    edge_distance_matrix = torch.nan_to_num(edge_distance_matrix, posinf=np.inf)

    # Step 2: Compute node-to-node shortest distances
    num_nodes = incidence_matrix.shape[0]
    shortest_distance_matrix = np.full((num_nodes, num_nodes), np.inf)

    # For each pair of nodes, calculate the shortest distance
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            # Find all hyperedges containing node i and node j
            edges_i = np.where(incidence_matrix[i, :].cpu().numpy() == 1)[0]
            edges_j = np.where(incidence_matrix[j, :].cpu().numpy() == 1)[0]

            # Compute the minimum distance between all pairs of edges from edges_i and edges_j
            min_distance = np.inf
            for e_i in edges_i:
                for e_j in edges_j:
                    min_distance = min(min_distance, edge_distance_matrix[e_i, e_j])

            # Add 1 to the minimum distance to account for the node-to-edge connection
            shortest_distance_matrix[i, j] = min_distance + 1
            shortest_distance_matrix[j, i] = shortest_distance_matrix[i, j]  # Symmetry

    # Convert the result to a PyTorch tensor
    shortest_distance = torch.tensor(shortest_distance_matrix, dtype=torch.float32)

    return shortest_distance

def shortest_edge_distances(real_incidence: torch.Tensor, combined_incidence: torch.Tensor, s=1):

    if not isinstance(real_incidence, torch.Tensor):
        real_incidence = torch.tensor(real_incidence, dtype=torch.float32)
    if not isinstance(combined_incidence, torch.Tensor):
        combined_incidence = torch.tensor(combined_incidence, dtype=torch.float32)

    R = real_incidence.shape[1]         # #pos
    A = combined_incidence.shape[1]     # #all
    V = A - R                           # #neg

    # ============ pos–pos ============ 
    real_adj = edge_adjacency_matrix(real_incidence, s)
    real_dist_np = dijkstra(
        real_adj.numpy(),
        directed=False,
        unweighted=True
    )
    real_dist = torch.from_numpy(real_dist_np).float()

    M_virt = torch.abs(combined_incidence[:, R:])
    M_real = torch.abs(real_incidence) 
    shared_count_vr = M_virt.T @ M_real
    adj_virt_real = (shared_count_vr >= s)

    dist_all = torch.full((A, A), float('inf'))
    dist_all[:R, :R] = real_dist
    for i in range(A):
        dist_all[i, i] = 0.

    # ============ pos–neg ============ 
    # dist(e, v) = min_{ e' in adj(v) } ( real_dist[e, e'] + 1 )
    for v_idx in range(V):
        v = R + v_idx
        e_neighbors = torch.where(adj_virt_real[v_idx])[0]
        if len(e_neighbors) == 0:
            continue
        min_dists_e = real_dist[:, e_neighbors].min(dim=1).values  # (R,)
        dist_e_v = min_dists_e + 1.
        
        dist_all[:R, v] = dist_e_v
        dist_all[v, :R] = dist_e_v

    # ============ neg-neg ============ 
    all_neighbors = []
    for v_idx in range(V):
        all_neighbors.append(torch.where(adj_virt_real[v_idx])[0])  # tensor of shape(?)

    for v1_idx in range(V):
        v1 = R + v1_idx
        e_neigh_1 = all_neighbors[v1_idx]
        if len(e_neigh_1) == 0:
            continue
        for v2_idx in range(V):
            if v1_idx == v2_idx:
                continue
            v2 = R + v2_idx
            e_neigh_2 = all_neighbors[v2_idx]
            if len(e_neigh_2) == 0:
                continue

            submatrix = real_dist[e_neigh_1][:, e_neigh_2]  # (len(e_neigh_1), len(e_neigh_2))
            min_dist_sub = submatrix.min().item()
            dist_v1_v2 = float('inf') if min_dist_sub == float('inf') else min_dist_sub + 2

            dist_all[v1, v2] = dist_v1_v2
            dist_all[v2, v1] = dist_v1_v2

    return dist_all

def diff_pooling(x, sign_h):
    return torch.mm(sign_h.T, x)