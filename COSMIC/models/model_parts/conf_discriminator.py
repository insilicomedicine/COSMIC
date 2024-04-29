import torch

from torch import nn

from .molgraph_embedding import MolGraphEmbedding

from torch_geometric.nn.models.schnet import GaussianSmearing, InteractionBlock

from torch_scatter import scatter_mean, scatter_sum

class ConfDiscriminator(nn.Module):
    def __init__(self,
                 node_size=128,
                 edge_size=32,
                 num_backbone_layers=3,
                 n_interactions=5,
                 num_gaussians=64,
                 conditions: str = 'none'):
        super(ConfDiscriminator, self).__init__()


        self.conditions = conditions

        self.bond_emb = nn.Sequential(
            GaussianSmearing(0.0, 10.0, num_gaussians))

        self.interactions = nn.ModuleList()
        for _ in range(n_interactions):
            block = InteractionBlock(node_size,
                                     edge_size + num_gaussians,
                                     edge_size + num_gaussians, 10.0)
            self.interactions.append(block)

        self.wgan_final_mlp = nn.Sequential(
            nn.Linear(node_size, node_size // 2),
            nn.LeakyReLU(),
            nn.Linear(node_size // 2, 1)
        )
        
        self.energy_final_mlp = nn.Sequential(
            nn.Linear(node_size, node_size // 2),
            nn.LeakyReLU(),
            nn.Linear(node_size // 2, 1)
        )

        self.molgraph_embedding = MolGraphEmbedding(node_size,
                                                    edge_size,
                                                    num_backbone_layers,
                                                    conditions=conditions)

    def forward(self, batch, cartesian_coords=None):
        node_emb, edge_emb = self.molgraph_embedding(batch)
        row, col = batch.edge_index
        edge_len = (cartesian_coords[row] - cartesian_coords[col]).norm(dim=-1)

        edge_emb = torch.cat([edge_emb, self.bond_emb(edge_len)], dim=-1)

        for interaction in self.interactions:
            node_emb = node_emb + interaction(node_emb,
                                              batch.edge_index,
                                              edge_len,
                                              edge_emb)

        return scatter_mean(self.wgan_final_mlp(node_emb), batch.batch, dim=0)[:, 0], scatter_sum(self.energy_final_mlp(node_emb), batch.batch, dim=0)[:, 0]