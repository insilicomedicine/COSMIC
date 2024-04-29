from typing import Union

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from collections import defaultdict

from energy_minimization.data.types import atom_to_idx, bond_to_idx, \
    charge_to_idx, NoStartTriplePoints, CollinearPointsException
from energy_minimization.utils import recursive_find_loop

from .coordinates_transforms import safe_norm


class SkeletonTransform(object):
    '''
    Performs a DFS on a molecule to prepare an atom order and a skeleton for
    computing a torsion coordinates.
    '''

    def __init__(self, walk_3d=False):
        self.walk_3d = walk_3d

    def process_one_mol(self,
                        data: Data) -> Data:
        row, col = data.edge_index

        adj_list = defaultdict(lambda: [])
        node_pair_to_edge_idx = defaultdict(lambda: None)

        for i, (st, end) in enumerate(
                zip(row.cpu().detach().numpy(),
                    col.cpu().detach().numpy())):
            adj_list[st].append(end)
            node_pair_to_edge_idx[(st, end)] = i

        was_here = None
        skeleton_mask = None
        order = None

        def dfs(cur_node_idx, depth=0):
            order[cur_node_idx] = max(order) + 1
            was_here[cur_node_idx] = True

            cur_neighs = 0

            next_nodes_list = np.random.permutation(
                [n for n in adj_list[cur_node_idx] if not was_here[n]])

            for next_node_idx in next_nodes_list:
                if not was_here[next_node_idx]:
                    skeleton_mask[
                        node_pair_to_edge_idx[(cur_node_idx, next_node_idx)]
                    ] = 1
                    skeleton_mask[
                        node_pair_to_edge_idx[(next_node_idx, cur_node_idx)]
                    ] = -1
                    cur_neighs += 1

                    if depth < 2 and cur_neighs > 1:
                        return False

                    if not dfs(next_node_idx, depth=depth + 1):
                        return False
            return True

        success = False

        if data.start_node_idx.shape[0]:
            start_node_candidates = data.start_node_idx.tolist()
        else:
            start_node_candidates = list(np.random.permutation(
                [i for i in range(data.num_nodes) if len(adj_list[i]) == 1]))

        for start_node_idx in start_node_candidates:
            was_here = [False for _ in range(data.num_nodes)]
            skeleton_mask = [0 for _ in range(data.num_edges)]
            order = [-1 for _ in range(data.num_nodes)]

            if dfs(cur_node_idx=start_node_idx):
                success = True
                break

        if not success:
            raise NoStartTriplePoints

        data_dict = {key: data[key] for key in data.keys}
        data_dict['skeleton_mask'] = \
            torch.tensor(skeleton_mask).to(data.x.device).long()
        data_dict['order'] = \
            torch.tensor(order).to(data.x.device).long()
        data_dict['start_node_idx'] = \
            torch.tensor(start_node_idx).to(data.x.device).long()

        new_data = Data.from_dict(data_dict)

        return new_data

    def __call__(self, data: Union[Data, Batch]) -> Union[Data, Batch]:
        if isinstance(data, Batch):
            data_list = data.to_data_list()

            processed_data = [self.process_one_mol(mol)
                              for mol in data_list]

            processed = Batch.from_data_list(processed_data)
        else:
            processed = self.process_one_mol(data)
        return processed


class AddStartPointTransform(object):
    '''
    Add start fictive point to molecules, where can't be found
    three starting points.
    '''

    def process_one_mol(self,
                        data: Data) -> Data:
        row, col = data.edge_index

        neighs = defaultdict(lambda: [])
        for (st, end) in zip(row.cpu().detach().numpy(),
                             col.cpu().detach().numpy()):
            neighs[st].append(end)

        hanging_nodes_idxs = [i for i in range(data.num_nodes)
                              if len(neighs[i]) == 1]
        ring_nodes = [i for i in range(data.num_nodes)
                      if data.is_in_ring[i].item()]

        first_node_candidates = \
            list(np.random.permutation(hanging_nodes_idxs)) + \
            list(np.random.permutation(ring_nodes))

        found = False
        for first_node_idx in first_node_candidates:
            second_node_candidates = neighs[first_node_idx]
            second_node_candidates = list(
                np.random.permutation(second_node_candidates))

            for second_node_idx in second_node_candidates:
                third_node_candidates = [i for i in neighs[second_node_idx]
                                         if i != first_node_idx]

                if len(third_node_candidates) == 0:
                    continue

                third_node_idx = np.random.choice(third_node_candidates, 1)[0]

                try:
                    u_12 = data.cartesian_coords[second_node_idx] - \
                           data.cartesian_coords[first_node_idx]
                    u_12 = u_12 / safe_norm(u_12, raise_expection=True)

                    u_23 = data.cartesian_coords[third_node_idx] - \
                           data.cartesian_coords[second_node_idx]
                    u_23 = u_23 / safe_norm(u_23, raise_expection=True)

                    n_123 = torch.cross(u_12, u_23)
                    n_123 = n_123 / safe_norm(n_123, raise_expection=True)

                    start_point_position = \
                        data.cartesian_coords[first_node_idx] - n_123

                    found = True
                except CollinearPointsException:
                    continue

                if found:
                    break
            if found:
                break

        if not found:
            raise NoStartTriplePoints

        start_atom_idx = data.num_nodes

        data_dict = {key: data[key] for key in data.keys}

        data_dict['start_node_idx'] = torch.LongTensor([start_atom_idx])
        data_dict['added_fictive_node'][0] = True

        data_dict['x'] = torch.cat(
            (data_dict['x'],
             torch.LongTensor([atom_to_idx['start']])),
            dim=0)
        data_dict['node_real'] = torch.cat(
            (data_dict['node_real'],
             torch.BoolTensor([False])),
            dim=0)
        data_dict['charge'] = torch.cat(
            (data_dict['charge'],
             torch.LongTensor([charge_to_idx['start']])),
            dim=0)

        data_dict['is_in_ring'] = torch.cat(
            (data_dict['x'],
             torch.LongTensor([0])),
            dim=0)

        data_dict['cartesian_coords'] = torch.cat(
            (data_dict['cartesian_coords'],
             start_point_position[None, :]),
            dim=0)
        data_dict['cartesian_y'] = torch.cat(
            (data_dict['cartesian_y'],
             start_point_position[None, :]),
            dim=0)

        data_dict['edge_index'] = torch.cat(
            (data_dict['edge_index'],
             torch.LongTensor(
                 [(start_atom_idx, first_node_idx),
                  (first_node_idx, start_atom_idx)]).transpose(0, 1)),
            dim=1)

        data_dict['edge_attr'] = torch.cat(
            (data_dict['edge_attr'],
             torch.LongTensor(
                 [bond_to_idx['start'], bond_to_idx['start']])),
            dim=0)

        data_dict['edge_real'] = torch.cat(
            (data_dict['edge_real'],
             torch.BoolTensor([False, False])),
            dim=0)

        new_data = Data.from_dict(data_dict)

        return new_data

    def __call__(self, data: Union[Data, Batch]) -> Union[Data, Batch]:
        if isinstance(data, Batch):
            data_list = data.to_data_list()

            processed_data = [self.process_one_mol(mol) for mol in
                              data_list]

            processed = Batch.from_data_list(processed_data)
        else:
            processed = self.process_one_mol(data)
        return processed

class RingTransform(object):
    def process_one_mol(self,
                        data: Data) -> Data:
        data_dict = {key: data[key] for key in data.keys}

        edge_index = data_dict['edge_index']
        edge_index_transformed, positions, _, _ = recursive_find_loop(
            edge_index=edge_index,
            positions=torch.tensor(y)
        )

        data_dict['edge_index_original'] = data_dict['edge_index'].long()
        data_dict['edge_index'] = edge_index_transformed.long()

        num_new_edges = (data_dict['edge_index'].size(1) -
                         data_dict['edge_index_original'].size(1))
        num_new_nodes = (positions.size(0) -
                         data_dict['cartesian_coords'].size(0))

        data_dict['edge_attr_original'] = data_dict['edge_attr']
        data_dict['edge_attr'] = torch.cat((
            data_dict['edge_attr'],
            torch.zeros((num_new_edges,), dtype=torch.long)
        ))
        data_dict['cartesian_coords'] = positions
        data_dict['cartesian_y'] = positions

        data_dict['x'] = torch.cat((
            data_dict['x'],
            torch.zeros((num_new_nodes,), dtype=torch.long)
        ))

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
