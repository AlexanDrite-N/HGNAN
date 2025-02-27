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
        self.feature_weights = nn.Parameter(torch.rand(self.in_channels))

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
        attention_weights = F.softmax(torch.exp(self.feature_weights), dim=0)
        fx_weighted = fx * attention_weights.unsqueeze(0).unsqueeze(-1)  # (N, num_features, out_channels)
        f_sums = fx_weighted.sum(dim=1)

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
        return stacked_results

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