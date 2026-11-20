"""Message-passing GNN layers and model, implemented in pure PyTorch.

Implements the update rule from the project proposal:

    h_i^(l+1) = sigma( W_self h_i^(l)  +  AGG({ W_neigh h_j^(l) : j in N(i) }) )

where AGG is mean aggregation (GraphSAGE-style) or content-based dot-product
attention (GAT-style). Aggregation is done over an edge_index tensor with
index_add_, so no compiled scatter kernels are required.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_sum(dst_index, values, num_nodes):
    if values.dim() == 1:
        out = torch.zeros(num_nodes, device=values.device, dtype=values.dtype)
        return out.index_add_(0, dst_index, values)
    out = torch.zeros(num_nodes, values.size(1),
                      device=values.device, dtype=values.dtype)
    return out.index_add_(0, dst_index, values)


def scatter_mean(dst_index, values, num_nodes):
    out = scatter_sum(dst_index, values, num_nodes)
    deg = torch.zeros(num_nodes, device=values.device)
    deg = deg.index_add_(0, dst_index,
                         torch.ones(dst_index.size(0), device=values.device))
    deg = deg.clamp(min=1).unsqueeze(1)
    return out / deg


def _neighbor_mask(edge_index):
    return edge_index[0] != edge_index[1]  # exclude self-loops from AGG


class SAGEConv(nn.Module):
    """Mean-aggregation layer: h_i' = W_self h_i + W_neigh * mean_{j in N(i)} h_j."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.neigh_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, edge_index):
        mask = _neighbor_mask(edge_index)
        src = edge_index[0][mask]
        dst = edge_index[1][mask]
        agg = scatter_mean(dst, self.neigh_lin(x)[src], x.size(0))
        return self.self_lin(x) + agg + self.bias


class DotAttnConv(nn.Module):
    """Content-based attention: weight each message by exp(h_src . h_dst / sqrt(d)),
    normalized over the receiving node's neighborhood (GAT-style, single head)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.scale = 1.0 / math.sqrt(out_dim)

    def forward(self, x, edge_index):
        h = self.lin(x)                    # n x out
        src = edge_index[0]
        dst = edge_index[1]
        e = (h[src] * h[dst]).sum(-1) * self.scale   # E
        e = e - e.max()                    # stabilize softmax
        e = torch.exp(e)
        denom = scatter_sum(dst, e, x.size(0))        # per-dest sum
        alpha = e / denom[dst]
        agg = scatter_sum(dst, h[src] * alpha.unsqueeze(1), x.size(0))
        return agg + self.bias


class GNN(nn.Module):
    """L-layer message-passing GNN with a per-node classification head."""

    def __init__(self, in_dim, hidden=64, layers=2, conv="sage", dropout=0.2):
        super().__init__()
        conv_cls = SAGEConv if conv == "sage" else DotAttnConv
        self.convs = nn.ModuleList()
        for i in range(layers):
            d_in = in_dim if i == 0 else hidden
            self.convs.append(conv_cls(d_in, hidden))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers - 1)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
        return self.head(x).squeeze(-1)   # logits, shape (n,)
