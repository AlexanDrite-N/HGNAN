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
            m_dist = self.m(distances.flatten().view(-1, 1))
            m_dist = m_dist.view(distances.size(0), distances.size(1), self.out_channels)

            if self.normalize_m:
                m_dist = m_dist / normalization_matrix.unsqueeze(-1)

            output = torch.sum(m_dist * f_sums.unsqueeze(0), dim=1)

        elif self.aggregation == "neighbor":
            N = distances.size(0)
            out_channels = f_sums.size(1)
            self_embedding = f_sums

            # distinguish neighbor(distances==0.5 because distances = 1/(real distances + 1))
            neighbor_mask = (distances == 0.5)

            neighbor_indices = neighbor_mask.nonzero(as_tuple=False)

            neighbor_agg = torch.zeros((N, out_channels), device=f_sums.device)
            neighbor_agg.index_add_(0, neighbor_indices[:, 0], f_sums[neighbor_indices[:, 1]])

            neighbor_counts = neighbor_mask.float().sum(dim=1, keepdim=True)
            avg_neighbors = torch.where(neighbor_counts > 0, neighbor_agg / neighbor_counts, torch.zeros_like(neighbor_agg))
            output = self_embedding + avg_neighbors

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