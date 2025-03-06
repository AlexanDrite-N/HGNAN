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

        # --------------------------------------------------
        # 构造 shape functions f_k
        # --------------------------------------------------
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
            # 利用邻接关系构造稀疏边索引： (i, j) 表示 j 是 i 的邻居
            neighbor_mask = (distances == 1.0)
            edge_indices = neighbor_mask.nonzero(as_tuple=False)  # 形状 (E, 2)

            if edge_indices.size(0) == 0:
                output = f_sums
            else:
                # 分别获得源节点和目标节点的索引
                i_indices = edge_indices[:, 0]  # 源节点索引
                j_indices = edge_indices[:, 1]  # 邻居节点索引

                # 对每条边计算 m(distances)，注意：边上的距离一般为1.0，但仍保持通用性
                edge_distances = distances[i_indices, j_indices].view(-1, 1)  # 形状 (E, 1)
                m_edge = self.m(edge_distances)  # 形状 (E, out_channels)

                # 如果需要归一化，则对每条边应用对应的归一化因子
                if self.normalize_m:
                    norm_values = normalization_matrix[i_indices, j_indices].view(-1, 1)
                    m_edge = m_edge / norm_values

                # 取出邻居 j 的特征向量
                f_j = f_sums[j_indices]  # 形状 (E, out_channels)

                # 每条边的贡献： m_edge * f_j
                edge_contrib = m_edge * f_j  # 形状 (E, out_channels)

                N = distances.size(0)
                out_channels = f_sums.size(1)
                # 初始化输出张量
                output = torch.zeros((N, out_channels), device=f_sums.device)
                # 利用 index_add_ 按源节点 i 聚合所有来自其邻居的贡献
                output.index_add_(0, i_indices, edge_contrib)

                # 统计每个节点的邻居数量
                neighbor_counts = torch.zeros((N, 1), device=f_sums.device)
                ones = torch.ones((edge_indices.size(0), 1), device=f_sums.device)
                neighbor_counts.index_add_(0, i_indices, ones)

                # 对于没有邻居的节点，直接用节点自身的特征
                mask = (neighbor_counts == 0).squeeze(1)
                output[mask] = f_sums[mask]

            # neighbor_mask = (distances == 1.0)

            # m_dist = self.m(distances.view(-1, 1)).view(distances.size(0), distances.size(1), -1)

            # if self.normalize_m:
            #     norm = normalization_matrix.unsqueeze(-1)
            #     m_dist = m_dist / norm

            # m_dist = m_dist * neighbor_mask.unsqueeze(-1).float()
            # output = torch.sum(m_dist * f_sums.unsqueeze(0), dim=1)

            # neighbor_counts = neighbor_mask.float().sum(dim=1, keepdim=True)
            # output = torch.where(neighbor_counts > 0, output, f_sums)
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