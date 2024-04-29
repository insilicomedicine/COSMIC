from typing import Union

import torch
from torch import nn

from torch_geometric.data import Batch, Data

from energy_minimization.data.types import atom_to_idx, bond_to_idx, \
    stereo_to_idx, charge_to_idx, chiral_to_idx
from .gcn_body import GCNNet


class MolGraphEmbedding(nn.Module):
    def __init__(self,
                 node_hidden_size: int = 256,
                 edge_hidden_size: int = 64,
                 num_layers: int = 6,
                 conditions: str = 'none',
                 lrelu_alpha: float = 0.01,
                 dropout: float = 0.0):
        super(MolGraphEmbedding, self).__init__()

        self.node_hidden_size = node_hidden_size
        self.edge_hidden_size = edge_hidden_size

        self.num_layers = num_layers

        self.conditions = conditions

        # embeddings
        self.atom_type_emb = nn.Embedding(len(atom_to_idx), node_hidden_size)
        self.chiral_tag_emb = nn.Embedding(len(chiral_to_idx), node_hidden_size)
        self.charge_emb = nn.Embedding(len(charge_to_idx), node_hidden_size)
        self.node_aggr = nn.Linear(3 * node_hidden_size, node_hidden_size)

        self.bond_type_emb = nn.Embedding(len(bond_to_idx), edge_hidden_size)
        self.stereo_type_emb = nn.Embedding(len(stereo_to_idx), edge_hidden_size)
        self.skeleton_emb = nn.Embedding(3, edge_hidden_size)
        self.past_future_emb = nn.Embedding(2, edge_hidden_size)
        self.edge_aggr = nn.Linear(4 * edge_hidden_size, edge_hidden_size)

        if self.conditions == 'descriptors3d':
            self.conditions_emb = nn.Sequential(
                nn.Linear(114, node_hidden_size // 2),
                nn.LeakyReLU(lrelu_alpha),
                nn.Linear(node_hidden_size // 2, node_hidden_size)
            )

        self.encoding_net = GCNNet(
            num_layers=num_layers,
            node_hidden_size=node_hidden_size,
            edge_hidden_size=edge_hidden_size,
            lrelu_alpha=lrelu_alpha,
            dropout=dropout,
            use_end_batchnorm=False)

    def forward(self, data: Union[Data, Batch]):
        past_future_mask = (data.order[data.edge_index[0]] <
                            data.order[data.edge_index[1]]).long()

        node_emb = self.node_aggr(torch.cat(
            [self.atom_type_emb(data.x.long()),
             self.charge_emb(data.charge.long()),
             self.chiral_tag_emb(data.chiral_tag.long())], dim=-1))
        edge_emb = self.edge_aggr(torch.cat(
            [self.bond_type_emb(data.edge_attr.long()),
             self.stereo_type_emb(data.stereo_type.long()),
             self.skeleton_emb(data.skeleton_mask.long() + 1),
             self.past_future_emb(past_future_mask.long())], dim=-1))
        edge_index = data.edge_index
        batch = data.batch

        if self.conditions == 'descriptors3d':
            node_emb = node_emb + self.conditions_emb(data.conditions)

        return self.encoding_net.forward_substituted(
            node_emb, edge_index, edge_emb, batch)
