import torch
import torch.nn.functional as F
from torch.nn import Linear, GRUCell
from torch_geometric.nn import aggr, GATv2Conv, TransformerConv, SAGPooling, MLP

from torch import Tensor
# from typing import Optional

class AGnLConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_heads: int = 4,
        edge_dim: int = -1,
        dropout: float = 0.0,
        conv_kwargs: dict = {},
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.edge_dim = edge_dim
        self.dropout = dropout

        if num_heads == 1:
            self.conv1 = TransformerConv(in_channels, hidden_channels, edge_dim=edge_dim,
                                   dropout=dropout, **conv_kwargs)
            self.gru1 = GRUCell(hidden_channels, in_channels)
            self.lin1 = Linear(in_channels, hidden_channels)
        elif num_heads >= 2:
            self.conv1 = TransformerConv(in_channels, hidden_channels//2, edge_dim=edge_dim,
                                   heads=num_heads, dropout=dropout, **conv_kwargs)
            self.lin0 = Linear(in_channels, hidden_channels*(num_heads//2))
            self.gru1 = GRUCell((hidden_channels//2)*num_heads, hidden_channels*(num_heads//2))
            self.lin1 = Linear(hidden_channels*(num_heads//2), hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_dim,
                                        add_self_loops=False, dropout=dropout, **conv_kwargs))
            self.grus.append(GRUCell(hidden_channels, hidden_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.gru1.reset_parameters()
        self.lin0.reset_parameters() if hasattr(self, 'lin0') else None
        self.lin1.reset_parameters()
        for conv, gru in zip(self.convs, self.grus):
            conv.reset_parameters()
            gru.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # Atom Embedding:
        if self.num_heads == 1:
            h = F.elu_(self.conv1(x, edge_index, edge_attr))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = self.gru1(h, x).relu_()
            x = F.leaky_relu_(self.lin1(x))
        elif self.num_heads >= 2:
            h = F.elu_(self.conv1(x, edge_index, edge_attr))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = F.leaky_relu_(self.lin0(x))
            x = self.gru1(h, x).relu_()
            x = F.leaky_relu_(self.lin1(x))
        g = [x]

        for conv, gru in zip(self.convs, self.grus):
            h = F.elu(conv(x, edge_index, edge_attr))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()
            g.append(x)

        return sum(g)   # sum hierarchical node embeddings

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'num_layers={self.num_layers}, '
                f'num_heads={self.num_heads}, '
                f'dropout={self.dropout})'
                f')')

class GnLTransformer(torch.nn.Module):
    def __init__(self, dim_in_G, dim_in_L, dim_h_conv, dim_h_lin, dim_out,
                 num_layer_conv, num_layer_lin, num_heads=4,
                 pool_k_G=20, pool_k_L=20, dropout=0.):
        super().__init__()
        # torch.manual_seed(42)
        self.conv_G = AGnLConv(in_channels=dim_in_G,
                                hidden_channels=dim_h_conv,
                                num_layers=num_layer_conv,
                                num_heads=num_heads,
                                dropout=dropout)
        self.conv_L = AGnLConv(in_channels=dim_in_L,
                                hidden_channels=dim_h_conv,
                                num_layers=num_layer_conv,
                                num_heads=num_heads,
                                dropout=dropout)

        self.pool_G = SAGPooling(dim_h_conv, ratio=pool_k_G)#, GNN=GATv2Conv)
        self.pool_L = SAGPooling(dim_h_conv, ratio=pool_k_L)#, GNN=GATv2Conv)
        self.sort_G = aggr.SortAggregation(k=pool_k_G)
        self.sort_L = aggr.SortAggregation(k=pool_k_L)
        self.pool_k_G = pool_k_G; self.pool_k_L = pool_k_L
        self.dropout = dropout

        self.mlp = MLP(in_channels=dim_h_conv*(pool_k_G+pool_k_L),
                       hidden_channels=dim_h_lin,
                       out_channels=dim_out,
                       num_layers=num_layer_lin,
                       dropout=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv_G.reset_parameters()
        self.conv_L.reset_parameters()
        self.pool_G.reset_parameters()
        self.pool_L.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, data_G, data_L):
        x_G, edge_index_G, edge_attr_G, batch_G = data_G.x, data_G.edge_index, data_G.edge_attr, data_G.batch
        x_L, edge_index_L, edge_attr_L, batch_L = data_L.x, data_L.edge_index, data_L.edge_attr, data_L.batch
        x_G = self.conv_G(x_G, edge_index_G, edge_attr_G)
        x_L = self.conv_L(x_L, edge_index_L, edge_attr_L)

        x_G, _, _, batch_G, _, _ = self.pool_G(x_G, edge_index_G, edge_attr_G, batch_G)
        x_L, _, _, batch_L, _, _ = self.pool_L(x_L, edge_index_L, edge_attr_L, batch_L)

        x_G = self.sort_G(x_G, batch_G)
        x_L = self.sort_L(x_L, batch_L)

        x = torch.cat([x_G, x_L], dim=1)
        x = self.mlp(x)

        return F.log_softmax(x, dim=1)