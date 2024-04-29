from typing import Union, Optional

import numpy as np
import torch
from torch import nn

from torch_geometric.data import Batch, Data
from torch_scatter import scatter_sum

from energy_minimization.utils.utils import add_prop
from energy_minimization.data.transforms import TorsionToCartesianTransform

from .molgraph_embedding import MolGraphEmbedding
from .gcn_body import GCNNet

EPS=1e-2

class ConfGenerator(nn.Module):
    def __init__(self,
                 latent_size: Optional[int] = None,
                 num_refiner_steps: int = 10,
                 conditions: str = 'none',
                 node_hidden_size: int = 256,
                 edge_hidden_size: int = 64,
                 num_backbone_layers: int = 6,
                 num_main_layers: int = 2):
        super(ConfGenerator, self).__init__()

        self.latent_size = latent_size

        self.node_hidden_size = node_hidden_size
        self.edge_hidden_size = edge_hidden_size

        self.num_backbone_layers = num_backbone_layers
        self.num_refiner_steps = num_refiner_steps

        self.conditions = conditions

        # nets
        self.molgraph_embedding = MolGraphEmbedding(node_hidden_size,
                                                    edge_hidden_size,
                                                    num_backbone_layers,
                                                    conditions)
        self.main_net = GCNNet(
            num_layers=num_main_layers,
            node_hidden_size=node_hidden_size,
            edge_hidden_size=edge_hidden_size,
            latent_size=latent_size)

        self.torsion_final_mlp = nn.Sequential(
            nn.Linear(node_hidden_size, node_hidden_size // 2),
            nn.LeakyReLU(),
            nn.Linear(node_hidden_size // 2, 3))

        self.refiner_final_mlp = nn.Sequential(
            nn.Linear(2 * node_hidden_size + edge_hidden_size, edge_hidden_size // 2),
            nn.LeakyReLU(),
            nn.Linear(edge_hidden_size // 2, 2))

    def forward(self, data: Union[Data, Batch]):
        edge_index = data.edge_index
        src, dest = data.edge_index
        batch = data.batch

        node_emb, edge_emb = self.molgraph_embedding(data)

        node_emb, edge_emb = self.main_net.forward_substituted(
            node_emb, edge_index, edge_emb, batch, data.latents)

        torsion_out = self.torsion_final_mlp(node_emb)

        rho = torch.norm(torsion_out, p=2, dim=-1).view(-1, 1)
        torsion_out = torsion_out / rho

        theta = torch.where(torch.abs(torsion_out[..., 0]) > torch.abs(torsion_out[..., 1]),
                            torch.atan2(torsion_out[..., 1], torsion_out[..., 0]),
                            np.pi / 2.0 - torch.atan2(torsion_out[..., 0], torsion_out[..., 1])).view(-1, 1)

        phi = torch.acos(torch.clamp(torsion_out[..., 2], min=-1+EPS, max=1-EPS)).view(-1, 1)

        torsion_preds = torch.cat([rho, phi, theta], dim=-1)

        refiner_out = self.refiner_final_mlp(torch.cat([edge_emb, node_emb[src], node_emb[dest]], dim=-1))
        edge_len = nn.Softplus()(refiner_out[:, :1] + 1.0)
        step_size = torch.sigmoid(refiner_out[:, 1:] - 2.0)

        add_prop(data, 'torsion_coords', torsion_preds, 'cartesian_y')
        data = TorsionToCartesianTransform()(data)
        cart_coords = data.cartesian_coords

        for _ in range(self.num_refiner_steps):

            start_pos, dest_pos = cart_coords[src], cart_coords[dest]
            force_dir = dest_pos - start_pos

            force_delta = edge_len - torch.norm(force_dir, p=2, dim=-1, keepdim=True)

            forces = 2 * step_size * force_delta * (force_dir / (torch.norm(force_dir, p=2, dim=-1, keepdim=True) + EPS))
            aggr_forces = scatter_sum(forces, dest, dim=0, dim_size=cart_coords.size(0))

            aggr_forces_norm = torch.norm(aggr_forces, p=2, dim=-1, keepdim=True) + EPS
            aggr_forces_dir = aggr_forces / aggr_forces_norm
            aggr_forces_clamped = aggr_forces_dir * \
                                  torch.clamp(aggr_forces_norm, max=0.5)

            cart_coords = cart_coords + aggr_forces_clamped

        return cart_coords, torsion_preds
