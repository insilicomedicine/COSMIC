from typing import Union

import torch
from torch_geometric.data import Data, Batch

from rdkit.Chem.rdmolops import RenumberAtoms
from rdkit import Chem

from ..types import chiral_to_idx, stereo_to_idx
from energy_minimization.utils import \
    compute_desciptors3d, descriptors3d_mean, descriptors3d_std


class StereometryFeaturesTransform(object):
    def __init__(self):
        pass

    def process_one_mol(self,
                        data: Data) -> Data:
        data_dict = {key: data[key] for key in data.keys}

        order = data_dict['order'].tolist()
        if data_dict['added_fictive_node'][0].item():
            order = [(i-1) for i in order[:-1]]

        shuffled_mol = RenumberAtoms(data_dict['mol'], order)

        chiral_tag = []
        stereo_type = []

        for i in range(data_dict['x'].shape[0]):
            if i < len(order):
                atom = shuffled_mol.GetAtomWithIdx(order[i])

                chiral_tag.append(chiral_to_idx[str(atom.GetChiralTag())])
            else:
                chiral_tag.append(chiral_to_idx['None'])

        row, col = data_dict['edge_index']
        for s, e in zip(row.tolist(), col.tolist()):
            if (s < len(order)) and (e < len(order)):
                bond = shuffled_mol.GetBondBetweenAtoms(order[s], order[e])

                if bond is not None:
                    stereo_type.append(stereo_to_idx[('forward', str(bond.GetStereo()))])
                else:
                    inv_bond = shuffled_mol.GetBondBetweenAtoms(order[e], order[s])
                    if inv_bond is not None:
                        stereo_type.append(stereo_to_idx[('backward', str(inv_bond.GetStereo()))])
                    else:
                        stereo_type.append(stereo_to_idx['None'])
            else:
                stereo_type.append(stereo_to_idx['None'])

        data_dict['stereo_type'] = torch.LongTensor(stereo_type)
        data_dict['chiral_tag'] = torch.LongTensor(chiral_tag)

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

class Descriptors3dTransform(object):
    def __init__(self):
        pass

    def process_one_mol(self,
                        data: Data) -> Data:
        data_dict = {key: data[key] for key in data.keys}

        curmol_descriptors = compute_desciptors3d(data_dict['mol'])
        curmol_descriptors = \
            (curmol_descriptors - torch.FloatTensor(descriptors3d_mean)) / \
            torch.FloatTensor(descriptors3d_std)

        curmol_descriptors = torch.clamp(curmol_descriptors, min=-3, max=3)
        curmol_descriptors = torch.nan_to_num(curmol_descriptors, nan=3.0, posinf=3.0, neginf=-3.)

        data_dict['conditions'] = curmol_descriptors[None, :].repeat(
            data_dict['x'].shape[0], 1)

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
