from typing import Union

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from energy_minimization.data.types import bond_to_idx, stereo_to_idx


class ShortestPathMatrixTransform(object):
    '''
    Computes a matrix of the shortest pathes between atoms via Ford-Bellman
    algo.
    '''

    def __init__(self):
        pass

    def process_one_mol(self, data: Data) -> Data:
        row, col = data.edge_index

        dist_mx = (data.num_nodes + 1) * np.ones(
            (data.num_nodes, data.num_nodes))
        dist_mx = dist_mx * (1 - np.eye(data.num_nodes))

        dist_mx[row.numpy(), col.numpy()] = 1

        for _ in range(data.num_nodes + 1):
            dist_mx = (dist_mx[:, :, None] + dist_mx[None, :, :]).min(axis=1)

        data_dict = {key: data[key] for key in data.keys}
        data_dict['shortest_path_mx'] = torch.tensor(dist_mx).flatten()

        new_data = Data.from_dict(data_dict)

        return new_data

    def __call__(self, data: Union[Data, Batch]) -> Union[Data, Batch]:
        if isinstance(data, Batch):
            data_list = data.to_data_list()

            processed_data = [self.process_one_mol(mol) for mol in data_list]

            processed = Batch.from_data_list(processed_data)
        else:
            processed = self.process_one_mol(data)

        return processed


class NeighborsEdgesTransform(object):
    '''
    Adds a fictive edges between n-neighborgs in the molecular graph.
    '''

    def __init__(self):
        pass

    def process_one_mol(self, data: Data) -> Data:
        dist_mx = data.shortest_path_mx.view(data.num_nodes,
                                             data.num_nodes)

        data_dict = {key: data[key] for key in data.keys}

        edge_index = data_dict['edge_index']
        edge_attr = data_dict['edge_attr']
        stereo_type = data_dict['stereo_type']
        edge_real = data_dict['edge_real']
        skeleton_mask = data_dict['skeleton_mask'] \
            if ('skeleton_mask' in data_dict) else None

        for d, edge_type in zip([2, 3],
                           ['SECOND_ORDER', 'THIRD_ORDER']):
            new_row, new_col = np.where(dist_mx == d)
            new_row, new_col = torch.tensor(new_row.copy()), torch.tensor(
                new_col.copy())

            new_edge_index = torch.stack((new_row, new_col), dim=0)

            edge_index = torch.cat((edge_index, new_edge_index), dim=-1)

            new_edge_attr = torch.LongTensor([
                bond_to_idx[edge_type]
                for _ in range(new_edge_index.shape[1])])
            edge_attr = torch.cat((edge_attr, new_edge_attr), dim=0)

            new_stereo_type = torch.LongTensor([
                stereo_to_idx['None']
                for _ in range(new_edge_index.shape[1])])
            stereo_type = torch.cat((stereo_type, new_stereo_type), dim=0)

            new_edge_real = torch.BoolTensor([
                False for _ in range(new_edge_index.shape[1])])
            edge_real = torch.cat((edge_real, new_edge_real), dim=0)

            if skeleton_mask is not None:
                new_skeleton_mask = torch.zeros_like(new_row).bool()
                skeleton_mask = torch.cat((skeleton_mask, new_skeleton_mask),
                                          dim=0)

        data_dict['edge_index'] = edge_index
        data_dict['edge_attr'] = edge_attr
        data_dict['stereo_type'] = stereo_type
        if skeleton_mask is not None:
            data_dict['skeleton_mask'] = skeleton_mask

        new_data = Data.from_dict(data_dict)

        return new_data

    def __call__(self, data: Union[Data, Batch]) -> Union[Data, Batch]:
        for key in ['shortest_path_mx']:
            if key not in data.keys:
                raise ValueError(f"Data doesn't contain {key}")

        if isinstance(data, Batch):
            data_list = data.to_data_list()

            processed_data = [self.process_one_mol(mol) for mol in data_list]

            processed = Batch.from_data_list(processed_data)
        else:
            processed = self.process_one_mol(data)

        return processed
