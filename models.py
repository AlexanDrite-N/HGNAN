import torch.nn as nn
import torch_geometric as pyg
import torch.nn.functional as F
import numpy as np
from utils import *

class HGNAM(nn.Module):
    def __init__(
          self,
          in_channels,
          out_channels,
          num_layers,
          hidden_channels=None,
          bias=True,
          dropout=0.0,
          device='cuda',
          limited_m=True,
          normalize_m=True,
          m_per_feature=False,
          weight = False,
          aggregation = "overall"
    ):
        
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.dropout = dropout
        self.limited_m = limited_m
        self.normalize_m = normalize_m
        self.m_per_feature = m_per_feature
        self.weight = weight
        self.aggregation = aggregation
        if self.weight == True:
            self.feature_weights = nn.Parameter(torch.rand(self.in_channels))

        # shape functions f_k
        self.fs = nn.ModuleList()
        for _ in range(in_channels):
            if num_layers == 1:
                layers = [nn.Linear(1, out_channels, bias=bias)]
            else:
                layers = [nn.Linear(1, hidden_channels, bias=bias), nn.ReLU(), nn.Dropout(p=dropout)]
                for _ in range(1, num_layers - 1):
                    layers += [nn.Linear(hidden_channels, hidden_channels, bias=bias), nn.ReLU(), nn.Dropout(p=dropout)]
                layers.append(nn.Linear(hidden_channels, out_channels, bias=bias))
            self.fs.append(nn.Sequential(*layers))

        # distance functions \rho
        if m_per_feature:
            self.ms = nn.ModuleList()
            for _ in range(out_channels if limited_m else in_channels):
                if num_layers == 1:
                    m_layers = [nn.Linear(1, out_channels, bias=bias)]
                else:
                    m_layers = [nn.Linear(1, hidden_channels, bias=bias), nn.ReLU()]
                    for _ in range(1, num_layers - 1):
                        m_layers += [nn.Linear(hidden_channels, hidden_channels, bias=bias), nn.ReLU()]
                    if limited_m:
                        m_layers.append(nn.Linear(hidden_channels, 1, bias=bias))
                    else:
                        m_layers.append(nn.Linear(hidden_channels, out_channels, bias=bias))
                self.ms.append(nn.Sequential(*m_layers))
        else:
            if num_layers == 1:
                m_layers = [nn.Linear(1, out_channels, bias=bias)]
            else:
                m_layers = [nn.Linear(1, hidden_channels, bias=bias), nn.ReLU()]
                for _ in range(1, num_layers - 1):
                    m_layers += [nn.Linear(hidden_channels, hidden_channels, bias=bias), nn.ReLU()]
                if limited_m:
                    m_layers.append(nn.Linear(hidden_channels, 1, bias=bias))
                else:
                    m_layers.append(nn.Linear(hidden_channels, out_channels, bias=bias))
            self.m = nn.Sequential(*m_layers)

    def forward(self, inputs):
        x, distances, normalization_matrix = inputs.x.to(self.device), inputs.dist_mat.to(self.device), inputs.norm_mat.to(self.device)
        fx = torch.empty(x.size(0), x.size(1), self.out_channels).to(self.device)
        for feature_index in range(x.size(1)):
            feature_col = x[:, feature_index].view(-1, 1)
            fx[:, feature_index] = self.fs[feature_index](feature_col)
        if self.weight == True:
            attention_weights = F.softmax(torch.exp(self.feature_weights), dim=0)
            fx_weighted = fx * attention_weights.unsqueeze(0).unsqueeze(-1)  # (N, num_features, out_channels)
            f_sums = fx_weighted.sum(dim=1)
        else:
            f_sums = fx.sum(dim=1)

        if self.aggregation == "overall":
            stacked_results = torch.empty(x.size(0), self.out_channels).to(self.device)
            for j, node in enumerate(range(x.size(0))):
                node_dists = distances[node] # Shape: (# nodes,)
                normalization = normalization_matrix[node] # Shape: (# nodes,)
                m_dist = self.m(node_dists.view(-1, 1))
                if self.normalize_m:
                    if m_dist.size(1) == 1:
                        m_dist = torch.div(m_dist, normalization.view(-1, 1)) # Shape: (# nodes, 1)
                    else:
                        for i in range(m_dist.size(1)):  # iterate number of classes
                            m_dist[:, i] = torch.div(m_dist[:, i], normalization) # Shape: (# nodes, # out_channel)
                pred_for_node = torch.sum(torch.mul(m_dist, f_sums), dim=0) # Shape: (# out_channels,)
                stacked_results[j] = pred_for_node.view(1, -1)
            output = stacked_results

        elif self.aggregation == "neighbor":
            # sparse edge index
            neighbor_mask = (distances == 1.0)
            edge_indices = neighbor_mask.nonzero(as_tuple=False)

            if edge_indices.size(0) == 0:
                output = f_sums
            else:
                # find target node and its neighbor's index
                i_indices = edge_indices[:, 0]
                j_indices = edge_indices[:, 1]

                # compute \rho (though it is trivial)
                edge_distances = distances[i_indices, j_indices].view(-1, 1)
                m_edge = self.m(edge_distances)

                if self.normalize_m:
                    norm_values = normalization_matrix[i_indices, j_indices].view(-1, 1)
                    m_edge = m_edge / norm_values

                # feature vector for neighbor
                f_j = f_sums[j_indices]

                # each neighbor's contribution: m_edge * f_j
                edge_contrib = m_edge * f_j

                N = distances.size(0)
                out_channels = f_sums.size(1)
                output = torch.zeros((N, out_channels), device=f_sums.device)
                # aggregation
                output.index_add_(0, i_indices, edge_contrib)

                # counting for #neighbor for each node
                neighbor_counts = torch.zeros((N, 1), device=f_sums.device)
                ones = torch.ones((edge_indices.size(0), 1), device=f_sums.device)
                neighbor_counts.index_add_(0, i_indices, ones)

                # in case that a node doesn't have neighbors, then use its own feature
                mask = (neighbor_counts == 0).squeeze(1)
                output[mask] = f_sums[mask]
        else:
            raise ValueError("Unknown aggregation type: {}".format(self.aggregation))
        return output

    def print_m_params(self):
        if hasattr(self, 'm'):
            print("Single m network parameters:")
            for name, param in self.m.named_parameters():
                print(name, param)
        elif hasattr(self, 'ms'):
            print("Separate m networks per dimension:")
            for idx, module in enumerate(self.ms):
                for name, param in module.named_parameters():
                    print(f"ms[{idx}].{name}", param)
        else:
            print("No m parameters found.")