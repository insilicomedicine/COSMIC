from typing import Union, Optional

import torch
from torch import nn

from torch_geometric.data import Batch, Data
from torch_geometric.nn.models.schnet import GaussianSmearing

from .molgraph_embedding import MolGraphEmbedding
from .gcn_body import GCNNet

class ConfEncoder(nn.Module):
    def __init__(self,
                 latent_size: Optional[int] = None,
                 num_backbone_layers: int = 6,
                 num_encoder_layers: int = 3,
                 node_hidden_size: int = 256,
                 edge_hidden_size: int = 64,
                 num_gaussians: int = 64,
                 conditions: str = 'none',
                 use_instance_norm: bool = True):
        super(ConfEncoder, self).__init__()

        self.latent_size = latent_size

        self.node_hidden_size = node_hidden_size
        self.edge_hidden_size = edge_hidden_size

        self.num_backbone_layers = num_backbone_layers
        self.num_encoder_layers = num_encoder_layers

        self.conditions = conditions

        self.molgraph_embedding = MolGraphEmbedding(node_hidden_size,
                                                    edge_hidden_size,
                                                    num_backbone_layers,
                                                    conditions)

        self.bond_len_emb = nn.Sequential(
            GaussianSmearing(start=0.0, stop=10.0, num_gaussians=num_gaussians),
            nn.Linear(num_gaussians, edge_hidden_size))

        self.encoder_net = GCNNet(
            num_layers=num_encoder_layers,
            node_hidden_size=node_hidden_size,
            edge_hidden_size=2 * edge_hidden_size,
            dropout=0.1,
            use_end_batchnorm=use_instance_norm)

        self.encoder_final_linear = nn.Sequential(
            nn.Linear(node_hidden_size, node_hidden_size // 2),
            nn.LeakyReLU(),
            nn.Linear(node_hidden_size // 2, latent_size)
        )

    def forward(self, data: Union[Data, Batch]):
        edge_index = data.edge_index
        src, dest = data.edge_index
        batch = data.batch

        bond_lens = torch.norm(data.cartesian_y[src] - data.cartesian_y[dest],
                               p=2, dim=-1)

        node_emb, edge_emb = self.molgraph_embedding(data)

        edge_emb_for_encoder = torch.cat([edge_emb, self.bond_len_emb(bond_lens)],
                                         dim=-1)

        node_emb = self.encoder_net.forward_substituted(
                node_emb,
                edge_index,
                edge_emb_for_encoder,
                batch)[0]

        latents = self.encoder_final_linear(node_emb)

        return latents
