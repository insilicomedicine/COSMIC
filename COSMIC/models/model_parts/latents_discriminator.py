from typing import Union, Optional

import torch
from torch import nn

from torch_geometric.data import Batch, Data
from torch_scatter import scatter_max, scatter_mean

from .molgraph_embedding import MolGraphEmbedding
from .gcn_body import GCNNet

class LatentDiscriminator(nn.Module):
    def __init__(self,
                 latent_size: Optional[int] = None,
                 num_backbone_layers: int = 6,
                 num_discriminator_layers: int = 3,
                 node_hidden_size: int = 256,
                 edge_hidden_size: int = 64,
                 conditions: str = 'none'):
        super(LatentDiscriminator, self).__init__()

        self.latent_size = latent_size

        self.node_hidden_size = node_hidden_size
        self.edge_hidden_size = edge_hidden_size

        self.num_discriminator_layers = num_discriminator_layers

        self.conditions = conditions

        # nets
        self.molgraph_embedding = MolGraphEmbedding(node_hidden_size,
                                                    edge_hidden_size,
                                                    num_backbone_layers,
                                                    conditions=conditions)

        self.discriminator_net = GCNNet(
            num_layers=num_discriminator_layers,
            node_hidden_size=node_hidden_size,
            edge_hidden_size=edge_hidden_size,
            latent_size=latent_size,
            dropout=0.1,
            lrelu_alpha=0.2)

        self.discriminator_final_mlp = nn.Sequential(
            nn.Linear(2 * node_hidden_size, node_hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(node_hidden_size, 1)
        )

    def forward(self, data: Union[Data, Batch]):
        edge_index = data.edge_index
        batch = data.batch

        node_emb, edge_emb = self.molgraph_embedding(data)

        discr_hidden = self.discriminator_net.forward_substituted(
            node_emb,
            edge_index,
            edge_emb,
            batch,
            data.latents)[0]

        discr_hidden = torch.cat([scatter_max(discr_hidden, data.batch, dim=0)[0],
                                  scatter_mean(discr_hidden, data.batch, dim=0)],
                                 dim=-1)

        return self.discriminator_final_mlp(discr_hidden).squeeze(-1)
