from typing import Union

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from energy_minimization.data.types import CollinearPointsException

from functools import partial

NORM_EPS = 1e-3
DOT_EPS = 1e-3


def safe_norm(t, dim=-1, keepdim=False, raise_expection=False):
    norm = torch.sqrt((t ** 2).sum(dim=dim, keepdim=keepdim))
    if raise_expection and norm.min() < NORM_EPS:
        raise CollinearPointsException
    norm = torch.clamp(norm, min=NORM_EPS)

    return norm


def get_u_base(cur_node_idx, cartesian, prev_node_idx_list,
               depth=1, norm=True):
    if depth == 1:
        u = cartesian[cur_node_idx] - \
            cartesian[prev_node_idx_list[cur_node_idx]]
        if norm:
            u = u / safe_norm(u, keepdim=True)
        return u
    else:
        return get_u_base(prev_node_idx_list[cur_node_idx],
                          cartesian, prev_node_idx_list, depth - 1,
                          norm)


def get_n_base(cur_node_idx, cartesian, prev_node_idx_list, depth=1,
               raise_expection=False):
    cross_product = torch.cross(
        get_u_base(cur_node_idx, cartesian, prev_node_idx_list, depth=depth),
        get_u_base(cur_node_idx, cartesian, prev_node_idx_list, depth=depth + 1),
        dim=-1)
    n = cross_product / safe_norm(cross_product, keepdim=True,
                                  raise_expection=raise_expection)
    return n


def dot_prod(a, b):
    return torch.clamp(torch.sum(a * b, dim=-1),
                       min=-1. + DOT_EPS, max=1. - DOT_EPS)


class CartesianToTorsionTransform(object):
    '''
    Performs computing torsion coordinates given cartesian coordinates.
    '''

    def __init__(self):
        pass

    def process(self, data: Data) -> Data:
        prev_node_idx_tensor = \
            -1 * torch.ones(data.x.shape[0],
                            dtype=torch.long,
                            device=data.cartesian_coords.device)
        sk_row, sk_col = data.edge_index[:, data.skeleton_mask == 1]
        prev_node_idx_tensor[sk_col] = sk_row

        torsion_coords = torch.zeros_like(data.cartesian_coords)

        get_u = partial(get_u_base,
                        cartesian=data.cartesian_coords,
                        prev_node_idx_list=prev_node_idx_tensor)

        get_n = partial(get_n_base,
                        cartesian=data.cartesian_coords,
                        prev_node_idx_list=prev_node_idx_tensor,
                        raise_expection=False)

        idx_range = torch.arange(data.num_nodes).long(). \
            to(torsion_coords.device)

        cross_product = torch.cross(
            get_n(idx_range, depth=2),
            get_u(idx_range, depth=2),
            dim=-1)

        dot_product = dot_prod(cross_product,
                               get_n(idx_range, depth=1))
        sign = torch.sign(dot_product)

        b_i = torch.norm(get_u(idx_range, depth=1, norm=False), dim=-1)
        a_i = torch.acos(dot_prod(-get_u(idx_range, depth=1),
                                  get_u(idx_range, depth=2)))
        cosine = dot_prod(
            -get_n(idx_range, depth=2),
            get_n(idx_range, depth=1))

        acos = torch.acos(cosine)
        d_i = sign * acos

        torsion_coords = torch.stack([b_i, a_i, d_i], dim=-1)

        # process first node in dfs order
        torsion_coords[data.order == 0, :] = 0.

        # process second node in dfs order
        torsion_coords[data.order == 1, 1] = np.pi
        torsion_coords[data.order == 1, 2] = 0

        # process third node in dfs order
        torsion_coords[data.order == 2, 2] = 0

        data_dict = {key: data[key] for key in data.keys}
        data_dict['torsion_coords'] = \
            torsion_coords.to(data.cartesian_coords.device)
        data_dict.pop('cartesian_coords')

        new_data = Data.from_dict(data_dict)

        return new_data

    def __call__(self, data: Union[Data, Batch]) -> Union[Data, Batch]:
        for key in ['cartesian_coords', 'skeleton_mask']:
            if key not in data.keys:
                raise ValueError(f"Data doesn't contain {key}")

        if (data.x.shape[0] != data.cartesian_coords.shape[0]) or (
                data.cartesian_coords.shape[1] != 3):
            raise ValueError(
                "data.cartesian_coords doesn't store 3d coordinates of atoms")

        processed = self.process(data)

        return processed


class TorsionToCartesianTransform(object):
    '''
        Performs computing cartesian coordinates given torsion coordinates.
    '''

    def __init__(self):
        pass

    def process(self, data: Data) -> Data:
        prev_node_idx_tensor = \
            -1 * torch.ones(data.x.shape[0],
                            dtype=torch.long,
                            device=data.torsion_coords.device)
        sk_row, sk_col = data.edge_index[:, data.skeleton_mask == 1]
        prev_node_idx_tensor[sk_col] = sk_row

        cartesian_coords = torch.zeros_like(data.torsion_coords)

        get_u = partial(get_u_base,
                        cartesian=cartesian_coords,
                        prev_node_idx_list=prev_node_idx_tensor)

        get_n = partial(get_n_base,
                        cartesian=cartesian_coords,
                        prev_node_idx_list=prev_node_idx_tensor)

        idx_range = torch.arange(data.num_nodes).long(). \
            to(cartesian_coords.device)

        for depth in range(data.order.max().item() + 1):
            cur_depth_idxs = idx_range[data.order == depth]

            b_i, a_i, d_i = data.torsion_coords[cur_depth_idxs].transpose(0, 1)

            if depth == 0:
                cartesian_coords[cur_depth_idxs, :] = 0
            elif depth == 1:
                cartesian_coords[cur_depth_idxs, 0] = b_i
                cartesian_coords[cur_depth_idxs, 1:] = 0
            else:
                # count vector coords
                first_coord = torch.cos(np.pi - a_i)
                second_coord = torch.sin(np.pi - a_i) * torch.cos(d_i)
                third_coord = torch.sin(np.pi - a_i) * torch.sin(d_i)

                # count cartesian coords in internal basis
                carterian_delta = b_i[:, None] * torch.stack(
                    [first_coord,
                     second_coord,
                     third_coord], dim=-1)

                if depth >= 3:
                    first_internal_cartesian_basis_vec = \
                        get_u(cur_depth_idxs, depth=2)
                    second_internal_cartesian_basis_vec = torch.cross(
                        get_n(cur_depth_idxs, depth=2),
                        get_u(cur_depth_idxs, depth=2)
                    )
                    third_internal_cartesian_basis_vec = \
                        get_n(cur_depth_idxs, depth=2)

                    # Transformation matrix
                    to_carterian_matrix = torch.stack(
                        [first_internal_cartesian_basis_vec,
                         second_internal_cartesian_basis_vec,
                         third_internal_cartesian_basis_vec], dim=-1)

                    # make transformation to main basis
                    carterian_delta = torch.bmm(to_carterian_matrix,
                                                carterian_delta.unsqueeze(
                                                    -1)).squeeze()

                cartesian_coords[cur_depth_idxs] = \
                    cartesian_coords[prev_node_idx_tensor[cur_depth_idxs]] + \
                    carterian_delta

        data_dict = {key: data[key] for key in data.keys}
        data_dict['cartesian_coords'] = \
            cartesian_coords.to(data.torsion_coords.device)
        data_dict.pop('torsion_coords')
        new_data = Data.from_dict(data_dict)

        return new_data

    def __call__(self, data: Union[Data, Batch]) -> Union[Data, Batch]:
        for key in ['torsion_coords', 'skeleton_mask']:
            if key not in data.keys:
                raise ValueError(f"Data doesn't contain {key}")

        if (data.x.shape[0] != data.torsion_coords.shape[0]) or \
                (data.torsion_coords.shape[1] != 3):
            raise ValueError(
                "data.torsion_coords doesn't store torsion coordinates of atoms")

        processed = self.process(data)

        return processed
