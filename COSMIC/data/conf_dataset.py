from typing_extensions import Literal

import os

from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

import json

from rdkit import Chem
from rdkit.Chem import SDMolSupplier

from .types import bond_to_idx, atom_to_idx,  \
    charge_to_idx, NoStartTriplePoints
from energy_minimization.data.transforms import \
    SkeletonTransform, CartesianToTorsionTransform, \
    ShortestPathMatrixTransform, NeighborsEdgesTransform, \
    AddStartPointTransform, Descriptors3dTransform, StereometryFeaturesTransform
from energy_minimization.utils import get_energy


def mol_to_data(mol, add_remote_neighs_edges,
                conditions, compute_energy, tries=10):
    """
    Transforms rdkit mol to pytorch_geometric Data object
    Args:
        :param: mol (rdkit.Chem.Mol) - rdkit mol object
        :param: undirected (bool) - if True - makes graph undirected
        :param: add_remote_neighs_edges (bool) - if True, add edges
            between not bonded atoms
        :param: add_ase_atoms (bool) - if True adds info about molecule
            for schnet
    Returns:
        pytorch_geometric Data object
    """
    conf = mol.GetConformer()
    y = np.array([np.array([conf.GetAtomPosition(atom_i).x,
                   conf.GetAtomPosition(atom_i).y,
                   conf.GetAtomPosition(atom_i).z])
         for atom_i in range(conf.GetNumAtoms())])

    node_features = []

    num_real_nodes = 0

    for a in mol.GetAtoms():
        if a.GetSymbol() in atom_to_idx:
            atom_type = atom_to_idx[a.GetSymbol()]
        else:
            atom_type = atom_to_idx['unknown']
        charge = charge_to_idx[a.GetFormalCharge()]
        is_in_ring = int(a.IsInRing())

        node_features.append({'atom_type': atom_type,
                              'charge': charge,
                              'is_in_ring': is_in_ring})
        num_real_nodes += 1

    adj_list = []
    edge_features = []
    neighs = defaultdict(lambda: [])
    for b in mol.GetBonds():
        s, e = sorted([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])
        bond_feat = {
            'bond_type': bond_to_idx[str(b.GetBondType())]}

        b1 = (s, e)
        b2 = (e, s)

        if s not in neighs[e]:
            neighs[e].append(s)

        if e not in neighs[s]:
            neighs[s].append(e)

        if b1 not in adj_list:
            adj_list.append(b1)
            edge_features.append(bond_feat)

        if b2 not in adj_list:
            adj_list.append(b2)
            edge_features.append(bond_feat)

    data_dict = {'x': torch.LongTensor([nf['atom_type']
                                        for nf in node_features]),
                 'node_real': torch.BoolTensor([True for _ in node_features]),
                 'num_real_nodes': num_real_nodes,
                 'charge': torch.LongTensor([nf['charge']
                                             for nf in node_features]),
                 'is_in_ring': torch.LongTensor([nf['is_in_ring']
                                                 for nf in node_features]),
                 'edge_index': torch.LongTensor(adj_list).transpose(0, -1),
                 'edge_attr': torch.LongTensor([ef['bond_type']
                                                for ef in edge_features]),
                 'edge_real': torch.BoolTensor([True for _ in edge_features]),
                 'cartesian_coords': torch.Tensor(y),
                 'cartesian_y': torch.Tensor(y),
                 'added_fictive_node': torch.BoolTensor([False]),
                 'start_node_idx': torch.LongTensor([]),
                 'mol': mol,
                 'num_heavy_atoms': Chem.RemoveHs(mol).GetNumAtoms(),
                 'conditions': torch.LongTensor([(0, )
                                                 for _ in node_features])}

    if compute_energy:
        data_dict['energy'] = get_energy(mol)

    #  transforms all loops in graph into stars until there is no loops in
    #  graph at all. Expands graph and node features with dummy atoms and
    #  edge features.
    data = Data.from_dict(data_dict)
    
    try:
        data = SkeletonTransform()(data)
    except NoStartTriplePoints:
        done = False
        for i in range(tries):
            try:
                cloned_data = data.clone()
                cloned_data = AddStartPointTransform()(cloned_data)
                cloned_data = SkeletonTransform()(cloned_data)
                data = cloned_data
                done = True
                break
            except NoStartTriplePoints:
                continue
        if not done:
            raise NoStartTriplePoints

    if conditions == 'descriptors3d':
        data = Descriptors3dTransform()(data)

    data = StereometryFeaturesTransform()(data)

    data = CartesianToTorsionTransform()(data)
    data_dict = {key: data[key] for key in data.keys}
    data_dict['torsion_y'] = data_dict['torsion_coords']
    data_dict['cartesian_coords'] = data_dict['cartesian_y']
    data = Data.from_dict(data_dict)

    data = ShortestPathMatrixTransform()(data)
    if add_remote_neighs_edges:
        data = NeighborsEdgesTransform()(data)

    return data


class ConfDataset(Dataset):
    def __init__(self,
                 root,
                 summary_path,
                 split=None,
                 subset='small',
                 transform=None,
                 pre_transform=None,
                 add_remote_neighs_edges: bool = True,
                 compute_energy: bool = False,
                 conditions: str = 'none',
                 task_type: Literal['distr_learn', 'argmin'] = 'distr_learn'):
        super(ConfDataset, self).__init__(root, transform, pre_transform)

        self.root = root

        self.add_remote_neighs_edges = add_remote_neighs_edges
        self.compute_energy = compute_energy

        self.conditions = conditions

        self.task_type = task_type

        self.items = []

        self.split = split
        self.subset = subset

        with open(summary_path, "r") as f:
            summary = json.load(f)

        for sm in summary.keys():
            if 'sdf_path' not in summary[sm]:
                continue

            if ('subsets' not in summary[sm]) or (
                    subset not in summary[sm]['subsets']):
                continue

            if (self.split is not None) and (
                    summary[sm]['split'] != self.split):
                continue

            if self.task_type == 'distr_learn':
                for i in range(summary[sm].get('uniqueconfs', 0)):
                    self.items.append((summary[sm]['sdf_path'], i))
            elif self.task_type == 'argmin':
                self.items.append((summary[sm]['sdf_path'], None))

    def len(self):
        return len(self.items)

    def get(self, idx):
        sdf_path, conf_id = self.items[idx]

        try:
            all_confs = SDMolSupplier(os.path.join(self.root, sdf_path))

            if self.task_type == 'distr_learn':
                conf = all_confs[conf_id]
            elif self.task_type == 'argmin':
                conf = min(all_confs,
                           key=lambda s: s.GetDoubleProp('totalenergy') if (
                                   s is not None) else 10 ** 100)

            conf = Chem.RemoveHs(conf)

            if (conf is None) or len(Chem.GetMolFrags(conf)) != 1:
                print(f'None conf or too few on idx {idx}')
                return self.get(np.random.randint(0, self.len()))

            data = mol_to_data(
                conf,
                add_remote_neighs_edges=self.add_remote_neighs_edges,
                conditions=self.conditions,
                compute_energy=self.compute_energy)
            return data
        except NoStartTriplePoints as e:
            print('Idx: ', idx, str(e))
            data = self.get(np.random.randint(0, self.len()))

        return data
