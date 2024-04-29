import torch
from torch import nn

from torch_geometric.nn import TransformerConv, GraphNorm
from torch_geometric.data import Batch, Data

from typing import Union, Optional


class EdgeResidualLayer(nn.Module):
    def __init__(self,
                 node_hidden_size: int,
                 edge_hidden_size: int,
                 dropout: float = 0.0):
        super(EdgeResidualLayer, self).__init__()
        self.node_hidden_size = node_hidden_size
        self.edge_hidden_size = edge_hidden_size

        self.change_mlp = nn.Sequential(
            nn.Linear(2 * node_hidden_size + edge_hidden_size, edge_hidden_size),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(edge_hidden_size, edge_hidden_size))

    def forward(self, x, edge_index, edge_attr):
        src, dest = edge_index

        return edge_attr + self.change_mlp(torch.cat([x[src], x[dest], edge_attr], -1))


class GCNBlock(nn.Module):
    def __init__(self,
                 node_hidden_size: int = 256,
                 edge_hidden_size: int = 64,
                 latent_size: Optional[int] = None,
                 dropout: float = 0.0,
                 lrelu_alpha: float = 0.01,
                 use_graphnorm: bool = False):
        super(GCNBlock, self).__init__()
        self.node_hidden_size = node_hidden_size
        self.bond_hidden_size = edge_hidden_size

        self.latent_size = latent_size
        if latent_size is not None:
            self.latent_emb = nn.Linear(latent_size, node_hidden_size)

        heads = node_hidden_size // 8

        self.node_updater = TransformerConv(
            in_channels=node_hidden_size,
            out_channels=node_hidden_size // heads,
            heads=heads,
            edge_dim=edge_hidden_size,
            dropout=dropout,
            bias=True)

        self.edge_updater = EdgeResidualLayer(
            node_hidden_size,
            edge_hidden_size,
            dropout)

        self.activation = nn.LeakyReLU(lrelu_alpha)

        self.use_graphnorm = use_graphnorm
        if use_graphnorm:
            self.graphnorm = GraphNorm(node_hidden_size)


    def forward_substituted(self,
                            x,
                            edge_index,
                            edge_attr,
                            batch,
                            latents=None):

        x = self.node_updater(
            x=x + (0 if (latents is None) else self.latent_emb(latents)),
            edge_index=edge_index,
            edge_attr=edge_attr)
        if self.use_graphnorm:
            x = self.graphnorm(x, batch)

        x = self.activation(x)

        edge_attr = self.edge_updater(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr)
        edge_attr = self.activation(edge_attr)

        return x, edge_attr

    def forward(self, data: Union[Data, Batch]):
        return self.forward_substituted(data.x,
                                        data.edge_index,
                                        data.edge_attr,
                                        data.batch,
                                        None if (self.latent_size is None) else data.latents)


class GCNNet(nn.Module):
    def __init__(self,
                 num_layers: int = 8,
                 node_hidden_size: int = 256,
                 edge_hidden_size: int = 64,
                 latent_size: Optional[int] = None,
                 dropout: float = 0.0,
                 use_end_batchnorm : bool = False,
                 lrelu_alpha: float = 0.01):
        super(GCNNet, self).__init__()

        self.node_hidden_size = node_hidden_size
        self.bond_hidden_size = edge_hidden_size
        self.latent_size = latent_size

        self.num_layers = num_layers

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.add_module(f'gcn_layer_{i}',
                                   GCNBlock(node_hidden_size,
                                            edge_hidden_size,
                                            latent_size,
                                            dropout,
                                            lrelu_alpha,
                                            use_end_batchnorm and (i == (num_layers-1)),
                                        ))

    def forward_substituted(self,
                            x,
                            edge_index,
                            edge_attr,
                            batch,
                            latents=None):
        for layer in self.layers:
            x, edge_attr = layer.forward_substituted(x,
                                                     edge_index,
                                                     edge_attr,
                                                     batch,
                                                     latents)

        return x, edge_attr

    def forward(self, data: Union[Data, Batch]):
        return self.forward_substituted(data.x,
                                        data.edge_index,
                                        data.edge_attr,
                                        data.batch,
                                        None if (self.latent_size is None) else data.latents)
