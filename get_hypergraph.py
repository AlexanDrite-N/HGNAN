from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
import os
import os.path as osp
import scipy.sparse as sp
import json
import pickle
import torch_geometric as pyg
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

from utils import *

def get_hypergraph(path='data/raw_data',
                   processed_data_dir='processed_data',dataset='iAF1260', 
                   train_size=0.6):
    """
    Constructs a hypergraph but focuses on hyperedge-level distance rather than node distance.
    1) Loads node features (optional).
    2) Loads a raw incidence matrix from a .mat file.
    3) Performs negative sampling to extend the incidence matrix with "negative" edges.
    4) Computes hyperedge-to-hyperedge distances + normalization.
    5) Splits edges into train/val/test sets (stratified).
    6) Packs data into a torch_geometric.data.Data object.
    7) Adds sign_h to indicate reactants/products for each hyperedge.
    """

    # ----------------------------
    # 1) Load node features
    # ----------------------------
    feature_path = f'./{path}/{dataset}/{dataset}.pt'
    node_features = torch.load(feature_path)
    # node_features: shape (N, in_channels)

    # ----------------------------
    # 2) Load raw incidence matrix from .mat
    # ----------------------------
    mat_path = f'./{path}/{dataset}/{dataset}.mat'
    data = loadmat(mat_path)
    H = data[dataset]['S']
    if isinstance(H, np.ndarray) and H.shape == (1, 1):
        H = H[0, 0]

    incidence_matrix = np.where(H > 0, 1, np.where(H < 0, 1, 0)).astype(np.float32)
    incidence_matrix = torch.tensor(incidence_matrix, dtype=torch.float)
    # ----------------------------
    # 3) Negative sampling
    # ----------------------------
    combined_H, num_neg_edges = negative_sampling(H) # combined_H shape: (#node, #pos_edges + #neg_edges)
    combined_H = torch.tensor(combined_H, dtype=torch.float)

    num_pos_edges = H.shape[1]
    total_edges = combined_H.shape[1]

    full_incidence_matrix = torch.tensor(np.where(combined_H > 0, 1, np.where(combined_H < 0, 1, 0)).astype(np.float32), dtype=torch.float)

    edge_features = diff_pooling(node_features, combined_H)
    print("Finish calculating edge embedding!")
    # ----------------------------
    # 4) Compute hyperedge-to-hyperedge distances & normalization
    # ----------------------------
    edge_dist = shortest_edge_distances(incidence_matrix, full_incidence_matrix, s=1)
    print("Finish calculating distances!")

    # Build a normalization matrix
    normalization_matrix = edge_dist.clone()
    for i, entry in enumerate(edge_dist):
        distances_counts = torch.unique(entry, return_counts=True)
        normalization_matrix[i].apply_(
            lambda x: distances_counts[1][(distances_counts[0] == x).nonzero().item()])

    # Optionally invert the distances (avoid zero-dist by adding 1)
    edge_dist = edge_dist.clone()
    edge_dist += 1
    edge_dist = 1.0 / edge_dist

    # ----------------------------
    # 5) Create labels & do train/val/test splits
    # ----------------------------
    labels = np.concatenate([np.ones(num_pos_edges), np.zeros(num_neg_edges)])
    assert len(labels) == total_edges, "Mismatch in label length"

    indices = np.arange(total_edges)
    test_size = 1 - train_size
    train_idx, test_idx, train_y, test_y = train_test_split(
        indices, labels, test_size=test_size, stratify=labels, random_state=0
    )

    # Build boolean masks
    train_mask = torch.zeros(total_edges, dtype=torch.bool)
    test_mask = torch.zeros(total_edges, dtype=torch.bool)

    train_mask[train_idx] = True
    test_mask[test_idx] = True

    # ----------------------------
    # 6) Pack everything into a Data object
    # ----------------------------
    hypergraph = Data(
        x=torch.tensor(edge_features, dtype=torch.float),  
        h=combined_H,                      
        y=torch.tensor(labels, dtype=torch.float),        
        dist_mat=edge_dist,                        
        norm_mat=normalization_matrix,           
        train_mask=train_mask,
        test_mask=test_mask
    )

    if not os.path.exists(f'{processed_data_dir}/{dataset}.pt'):
      torch.save(hypergraph, f'{processed_data_dir}/{dataset}.pt')
      print(f'Saved preprocessed {dataset} dataset')

    return hypergraph

def load_citation_dataset(path='data/raw_data', processed_data_dir='processed_data', data_name='cora', train_size=0.5, val_size=0.25):
    '''
    This function reads the citation dataset from HyperGCN and converts the hypergraph
    into an incidence matrix where rows represent nodes and columns represent hyperedges.
    '''

    print(f'Loading hypergraph dataset from hyperGCN: {data_name}')

    # Load node features
    with open(osp.join(path, data_name, 'features.pickle'), 'rb') as f:
        features = pickle.load(f)
        features = features.todense()

    # Load node labels
    with open(osp.join(path, data_name, 'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'Number of nodes: {num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # Load hypergraph
    with open(osp.join(path, data_name, 'hypergraph.pickle'), 'rb') as f:
        hypergraph = pickle.load(f)  # { hyperedge: [list of nodes in the he], ...}

    num_hyperedges = len(hypergraph)

    # Create incidence matrix H
    H = np.zeros((num_nodes, num_hyperedges), dtype=np.float32)

    for he_idx, (he, nodes) in enumerate(hypergraph.items()):
        for node in nodes:
            H[node, he_idx] = 1

    # Convert H to a sparse tensor
    H = torch.FloatTensor(H)
    non_zero_count = np.count_nonzero(H)

    node_distances = shortest_node_distances(H, s=1)

    normalization_matrix = node_distances.clone()
    for i, entry in enumerate(node_distances):
        distances_counts = torch.unique(entry, return_counts=True)
        normalization_matrix[i] = normalization_matrix[i].cpu().apply_(
            lambda x: distances_counts[1][(distances_counts[0].float() == x).nonzero().item()]
        )

    node_distances = 1 / node_distances

    labels = labels[:num_nodes]

    indices = np.arange(num_nodes)
    train_val_size = train_size + val_size
    test_size = 1 - train_val_size
    val_ratio = val_size / train_val_size

    train_val_idx, test_idx, train_val_y, test_y = train_test_split(
        indices, labels, test_size=test_size, stratify=labels, random_state=0
    )

    train_idx, val_idx, train_y, val_y = train_test_split(
        train_val_idx, train_val_y, test_size=val_ratio, stratify=train_val_y, random_state=0
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # Construct PyTorch Geometric Data object
    data = Data(x=features, 
                y=labels, 
                h=H,
                dist_mat = torch.tensor(node_distances, dtype=torch.float).squeeze(0),
                norm_mat = torch.tensor(normalization_matrix, dtype=torch.float).squeeze(0),
                train_mask=train_mask, 
                val_mask=val_mask,
                test_mask=test_mask)

    if not os.path.exists(f'{processed_data_dir}/{data_name}.pt'):
        torch.save(data, f'{processed_data_dir}/{data_name}.pt')
        print(f'Saved preprocessed {data_name} dataset')
    return data

def load_LE_dataset(path='data/raw_data', processed_data_dir='processed_data', data_name="Mushroom", train_size=0.5, val_size=0.25):
    # load edges, features, and labels.
    print('Loading {} dataset...'.format(data_name))

    file_name = f'{data_name}.content'
    p2idx_features_labels = osp.join(path, data_name, file_name)
    idx_features_labels = np.genfromtxt(p2idx_features_labels,dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))
    if data_name == 'zoo':
        labels = labels - 1 

    print('load features')

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    file_name = f'{data_name}.edges'
    p2edges_unordered = osp.join(path, data_name, file_name)
    edges_unordered = np.genfromtxt(p2edges_unordered,
                                    dtype=np.int32)
    print(f'edges_unordered: {edges_unordered}')
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
    output_file = "edges.txt"
    np.savetxt(output_file, edges, fmt='%d', delimiter=' ')
    print('load edges')
    print(f"edges:{edges}")

    # From adjacency matrix to edge_list
    edge_index = edges.T
    assert edge_index[0].max() == edge_index[1].min() - 1

    # check if values in edge_index is consecutive. i.e. no missing value for node_id/he_id.
    assert len(np.unique(edge_index)) == edge_index.max() + 1

    num_nodes = edge_index[0].max() + 1
    num_hyperedges = edge_index[1].max() - num_nodes + 1

    edge_index = np.hstack((edge_index, edge_index[::-1, :]))

    H = torch.zeros((num_nodes, num_hyperedges), dtype=torch.float32)
    print(f'H.shape:{H.shape}')

    for i in range(edge_index.shape[1]//2):
        v = edge_index[0, :][i]
        e = edge_index[1, :][i] - num_nodes
        H[v, e] = 1
    print(f"num_class:{len(np.unique(labels[:num_nodes].numpy()))}")

    features = torch.FloatTensor(np.array(features[:num_nodes].todense()))

    node_distances = shortest_node_distances(H, s=1)
    print(f"node_distances: {node_distances}")
    normalization_matrix = node_distances.clone()
    for i, entry in enumerate(node_distances):
        distances_counts = torch.unique(entry, return_counts=True)
        # Move the tensor to CPU temporarily for applying the operation
        normalization_matrix[i] = normalization_matrix[i].cpu().apply_(
            lambda x: distances_counts[1][(distances_counts[0].float() == x).nonzero().item()]
        )  # Move back to the original device

    node_distances = 1 / node_distances

    print(f"normalization_matrix:{normalization_matrix}")

    labels = labels[:num_nodes]

    indices = np.arange(num_nodes)
    train_val_size = train_size + val_size
    test_size = 1 - train_val_size
    val_ratio = val_size / train_val_size

    train_val_idx, test_idx, train_val_y, test_y = train_test_split(
        indices, labels, test_size=test_size, stratify=labels, random_state=42
    )

    train_idx, val_idx, train_y, val_y = train_test_split(
        train_val_idx, train_val_y, test_size=val_ratio, stratify=train_val_y, random_state=42
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # build torch data class
    data = Data(x=features, 
                y=labels,
                h=H,
                dist_mat = torch.tensor(node_distances, dtype=torch.float).squeeze(0), 
                norm_mat = torch.tensor(normalization_matrix, dtype=torch.float).squeeze(0),
                train_mask=train_mask, 
                val_mask=val_mask,
                test_mask=test_mask)
    
    if not os.path.exists(f'{processed_data_dir}/{data_name}.pt'):
        torch.save(data, f'{processed_data_dir}/{data_name}.pt')
        print(f'Saved preprocessed {data_name} dataset')
    return data