from functools import partial

import torch
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem

import networkx as nx

from copy import deepcopy

from rdkit.Chem import rdMolDescriptors, rdMolTransforms

from scipy.linalg import svd, det
import numpy as np

descriptors3d_mean = [1.38561939e+01, 4.17437228e+00, 1.71468547e+00, 6.80266842e-01,
       2.25536487e-01, 1.50661409e-01, 1.50799707e-01, 1.51311665e-01,
       5.17661391e-01, 4.50929585e-01, 4.14941465e-01, 1.25776776e+01,
       3.48021593e+00, 1.23253243e+00, 7.03641112e-01, 2.17618902e-01,
       1.50659237e-01, 1.50799661e-01, 1.51306646e-01, 4.48169698e-01,
       3.15021062e-01, 1.95750020e-01, 1.29685609e+01, 3.66456341e+00,
       1.37024757e+00, 6.97630865e-01, 2.18985338e-01, 1.50658545e-01,
       1.50801063e-01, 1.51318101e-01, 4.53972621e-01, 3.44593597e-01,
       2.55422598e-01, 1.37360948e+01, 4.13945605e+00, 1.69807546e+00,
       6.80177756e-01, 2.25678839e-01, 1.50660578e-01, 1.50800180e-01,
       1.51312206e-01, 5.08892653e-01, 4.44258048e-01, 4.06630408e-01,
       1.32482460e+01, 3.79065412e+00, 1.44651084e+00, 6.94365604e-01,
       2.20141439e-01, 1.50659549e-01, 1.50800133e-01, 1.51317403e-01,
       4.73712667e-01, 3.68497359e-01, 2.86945066e-01, 1.39395625e+01,
       4.23896301e+00, 1.76059876e+00, 6.77754356e-01, 2.26575792e-01,
       1.50660127e-01, 1.50799306e-01, 1.51308599e-01, 5.24632505e-01,
       4.66183081e-01, 4.38872933e-01, 1.33359800e+01, 4.03125155e+00,
       1.61701536e+00, 6.80361464e-01, 2.26844602e-01, 1.50657867e-01,
       1.50799783e-01, 1.51306500e-01, 5.00332437e-01, 4.29709107e-01,
       3.49088902e-01, 1.97452517e+01, 1.72904264e+01, 1.80033719e+01,
       1.95736267e+01, 1.84854103e+01, 1.99391243e+01, 1.89842467e+01,
       8.57714180e+01, 6.06815926e+01, 6.73857124e+01, 8.42332529e+01,
       7.18963061e+01, 8.81018841e+01, 7.89607532e+01, 1.50926231e-01,
       1.50924488e-01, 5.27813090e-01, 5.63324678e-01, 5.54132695e-01,
       5.27704697e-01, 5.49180078e-01, 5.24026189e-01, 5.28102934e-01,
       4.61940737e-01, 3.20410110e-01, 3.52093080e-01, 4.54024015e-01,
       3.77148663e-01, 4.77325696e-01, 4.27140118e-01, 2.00170394e+02,
       1.29016296e+02, 1.47422449e+02, 1.95855454e+02, 1.59700286e+02,
       2.07304054e+02, 1.80372042e+02]

descriptors3d_std = [6.07778008e+00, 1.56111965e+00, 7.09514223e-01, 1.17487661e-01,
       8.92395182e-02, 7.29302189e-03, 7.22559045e-03, 1.42353836e-02,
       6.54616962e-02, 6.01991183e-02, 1.07171913e+00, 5.76706697e+00,
       1.41404263e+00, 6.15664480e-01, 1.21937440e-01, 9.45682454e-02,
       7.21736498e-03, 7.22745411e-03, 1.41336864e-02, 1.59800904e-01,
       1.10143865e-01, 1.07311207e+00, 5.80160407e+00, 1.46817268e+00,
       6.39150486e-01, 1.20778200e-01, 9.32880874e-02, 7.23523067e-03,
       7.20207723e-03, 1.42569292e-02, 7.40772666e-02, 6.81700250e-02,
       1.07181478e+00, 6.03212147e+00, 1.54333201e+00, 7.03068900e-01,
       1.17590399e-01, 8.92657130e-02, 7.28272856e-03, 7.21113850e-03,
       1.42261998e-02, 6.52734519e-02, 6.08103246e-02, 1.07180258e+00,
       5.88756890e+00, 1.50274965e+00, 6.58120479e-01, 1.20057783e-01,
       9.25038998e-02, 7.24912775e-03, 7.21095076e-03, 1.42527443e-02,
       7.39092916e-02, 6.73332810e-02, 1.07177214e+00, 6.10172174e+00,
       1.56752728e+00, 7.17819703e-01, 1.17053539e-01, 8.86862102e-02,
       7.26795292e-03, 7.22770247e-03, 1.42097836e-02, 6.69747497e-02,
       6.28404086e-02, 1.07185502e+00, 5.93728018e+00, 1.48808680e+00,
       6.93158600e-01, 1.17846231e-01, 8.91450919e-02, 7.21201674e-03,
       7.23766373e-03, 1.41960217e-02, 1.38613043e-01, 1.25475136e-01,
       1.07597416e+00, 5.82301836e+00, 5.46543374e+00, 5.51025080e+00,
       5.77072130e+00, 5.60486655e+00, 5.84966467e+00, 5.67275387e+00,
       3.93939402e+01, 2.98122093e+01, 3.22982952e+01, 3.85783631e+01,
       3.42039901e+01, 4.01701552e+01, 3.63963261e+01, 7.73453584e-03,
       7.73254659e-03, 1.66656613e-01, 1.72442577e-01, 1.70941327e-01,
       1.66792809e-01, 1.69980985e-01, 1.66081624e-01, 1.67149889e-01,
       1.07046982e+00, 1.07173509e+00, 1.07053491e+00, 1.07048941e+00,
       1.07053610e+00, 1.07050909e+00, 1.07160214e+00, 1.05887199e+02,
       7.42046036e+01, 8.33039788e+01, 1.03437108e+02, 8.94725829e+01,
       1.08532451e+02, 9.55581206e+01]

def add_prop(batch_or_data, prop_name, prop, as_prop='cartesian_y'):
    if isinstance(batch_or_data, Batch):
        if prop_name not in batch_or_data.keys:
            batch_or_data.keys.append(prop_name)
            batch_or_data._slice_dict[prop_name] = batch_or_data._slice_dict[as_prop]
            batch_or_data._inc_dict[prop_name] = batch_or_data._inc_dict[as_prop]
        batch_or_data[prop_name] = None
        batch_or_data[prop_name] = prop
    elif isinstance(batch_or_data, Data):
        data_dict = {key: batch_or_data[key] for key in batch_or_data.keys}
        data_dict['prop_name'] = prop
        batch_or_data = Data.from_dict(data_dict)

    return batch_or_data


def compute_alligment(A_pc, B_pc):
    '''
    Based on http://nghiaho.com/?page_id=671
    '''
    A = A_pc.T
    B = B_pc.T

    A_centered = A - torch.mean(A, dim=1, keepdim=True)
    B_centered = B - torch.mean(B, dim=1, keepdim=True)

    H = (B_centered @ A_centered.T).detach().cpu().numpy()

    try:
        U, _, V = svd(H + 0.05 * np.eye(H.shape[0]))
    except:
        U, _, V = svd(np.eye(H.shape[0]))

    R = V.T @ U.T

    if np.isnan(R).any() or np.abs(det(R)) > 10:
        R = np.eye(H.shape[0])

    R = torch.tensor(R).float().to(B.device)

    B_pc_new = R @ B_centered + torch.mean(A, dim=1, keepdim=True)
    B_pc_new = B_pc_new.T

    return A_pc, B_pc_new.detach()


def get_energy(mol, normalize=True, addHs=True):
    try:
        if addHs:
            mol_to_compute = Chem.AddHs(mol, addCoords=True)
        else:
            mol_to_compute = mol
        
        prop = AllChem.MMFFGetMoleculeProperties(mol_to_compute)
        ff = AllChem.MMFFGetMoleculeForceField(mol_to_compute, prop)
        energy = ff.CalcEnergy()
        if normalize:
            energy = energy / Chem.RemoveHs(mol).GetNumAtoms()

            if np.isnan(energy) or energy > 100.:
                energy = 100.
    except:
        if normalize:
            energy = 100.
        else:
            energy = 10000.

    return energy


def set_conformer(mol, coords):
    new_mol = Chem.Mol(mol)
    conf = new_mol.GetConformer()

    for i in range(new_mol.GetNumAtoms()):
        conf.SetAtomPosition(i, coords[i].detach().cpu().tolist())

    return new_mol


def find_bond_in_template(template_bonds, bond):
    """
    Template_bonds - list of bonds (for example - ring)
        bond - bond, which we are searching in template_bond
    Returns True if template_bonds contains bond
        and False otherwise
    """

    return bond not in template_bonds


def transform_ring(edge_index, ring):
    """
    Function for transforming transforming ring
        if input graph to 'star'.
    edge_index (torch.tensor)
    ring - tuple or list
    Returns updated edge indieces list
    """

    # add metanode, which connected with each ring's atom
    list_edge_index = deepcopy(edge_index)
    new_edge = torch.max(list_edge_index) + 1
    list_edge_index = list_edge_index.tolist()

    new_structure = [new_edge.item()] * len(ring) + list(map(float, ring))
    new_structure = [new_structure]
    new_structure.append(list(reversed(new_structure[0])))
    list_edge_index = [list_edge_index[i] + new_structure[i] for i in [0, 1]]
    # destruct ring bonds
    bond_ring = [[ring[i], ring[(j)]]
                 for i in range(len(ring)) for j in range(len(ring)) if i != j]
    find_ring = partial(find_bond_in_template, bond_ring)
    list_edge_index_transpose = np.array(list_edge_index).T.tolist()
    mask = list(map(find_ring, list_edge_index_transpose))
    return (torch.tensor(list_edge_index_transpose)[mask]).permute(1, 0)


def get_ring_point(ring, positions):
    """
    Returns center point of the ring
    """

    dummy_positions = torch.mean(positions[ring], dim=0)
    return dummy_positions


def ring_star_transformation(edge_index, rings,
                             positions,
                             rings_centers_info=None):
    """
    Transforms all rings in input graph to 'stars'
        edge_index (torch.tensor)
        rings - list of tuples or lists
        positions - np.array
        rings_centers_info - indieces of dummy atoms (ring centers)
    Returns:
        edge_index - updated graph
        positions - updated positions of atoms with positions of ring centers
        rings_centers_info - updated indieces of dummy atoms (ring centers)
    """

    dummy_atoms_pos = []
    if rings_centers_info is None:
        rings_centers_info = []
    for ring_num, ring in enumerate(rings):
        edge_index = transform_ring(edge_index, ring)
        rings_centers_info.append(int(torch.max(edge_index)))
        dummy_atoms_pos.append(get_ring_point(ring, positions))
    if len(dummy_atoms_pos) != 0:
        dummy_atoms_pos = torch.stack(dummy_atoms_pos)
        positions = torch.cat((positions, dummy_atoms_pos))
    return edge_index, positions, rings_centers_info


def find_loops(edge_index):
    """
    Finds all loops with len > 2 and < 8 in input graph
    Returns List[tuple] - all rings in graph
    """

    G = nx.Graph()
    for i in range(edge_index.size(1)):
        G.add_edge(edge_index[0][i].item(), edge_index[1][i].item())
    DG = nx.DiGraph(G)
    nx.simple_cycles(DG)
    rings = list(set([frozenset(loop) for loop in list(nx.simple_cycles(DG))
                      if (len(loop) > 2 and len(loop) < 8)]))
    loops_remove = []
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(rings[i].intersection(
                    rings[j])) >= 3:  # rings cannot intersect this way
                assert len(rings[i]) != len(rings[j])
                if len(rings[i]) > len(rings[j]):
                    loops_remove.append(i)
                elif len(rings[i]) < len(rings[j]):
                    loops_remove.append(j)

    rings = [list(rings[i]) for i in range(len(rings)) if i not in loops_remove]
    return rings


def recursive_find_loop(edge_index, rings=None,
                        positions=None, num_loops=0,
                        rings_centers_info=None):
    '''
    Returns:
        When made ring-star transformation in neighbour loops - new loops can be
        produced - so, run this algorithm recursively, while no loops in graph.
        (All input molecules are not contain more then 2 neighbour
         loops with each other)
        edge_index_transformed - adj list with transformed rings to stars
        positions - torch.tensor of atom and fictive atoms (rings centers)
            positions
        num_loops - number of loops
        rings_centers_info - indieces of nodes which actually are ring
             centers

    '''
    if rings is None:
        rings = []
    if rings_centers_info is None:
        rings_centers_info = []
    edge_index, positions, rings_centers_info = ring_star_transformation(
        edge_index, rings, positions, rings_centers_info
    )
    rings = find_loops(edge_index)
    if len(rings) != 0:
        num_loops += len(rings)
        edge_index, positions, num_loops, rings_centers_info = \
            recursive_find_loop(
                edge_index, rings,
                positions, num_loops,
                rings_centers_info=rings_centers_info
            )
    return edge_index, positions, num_loops, rings_centers_info


def compute_desciptors3d(mol):
    mol_to_compute = Chem.AddHs(mol, addCoords=True)
    rdMolTransforms.CanonicalizeMol(mol_to_compute, ignoreHs=False)
    
    return torch.FloatTensor(np.clip(rdMolDescriptors.CalcWHIM(mol_to_compute, 0), -1000, 1000))

def twice_batch(batch):
    data_list = batch.to_data_list()
    twiced_data_list = data_list + data_list
    return Batch.from_data_list(twiced_data_list)


def reconstruction_loss(batch, torsion_pred, cartesian_pred):
    losses_dist_mx = []
    losses_cos = []

    add_prop(batch, 'torsion_pred', torsion_pred)
    add_prop(batch, 'cartesian_pred', cartesian_pred)

    for data in batch.to_data_list():
        loss_d = ((data.torsion_y[:, 0] - data.torsion_pred[:, 0]) ** 2).mean()

        cur_torsion_cos_loss = 1 - torch.cos(data.torsion_y[:, 1] - data.torsion_pred[:, 1]).mean()
        cur_dihedreal_cos_loss = 1 - torch.max(
            torch.stack([(torch.cos(data.torsion_y[:, 2] - sign * data.torsion_pred[:, 2])).mean()
                         for sign in [1., -1.]])) # cover anantiomers

        cur_cos_loss = loss_d + cur_torsion_cos_loss + cur_dihedreal_cos_loss

        losses_cos.append(cur_cos_loss)

        # compute dist mx loss
        pair_mx_pred = torch.norm(
            data.cartesian_pred[:, None, :] -
            data.cartesian_pred[None, :, :],
            dim=-1)
        pair_mx_true = torch.norm(data.cartesian_y[:, None, :] -
                                  data.cartesian_y[None, :, :],
                                  dim=-1)

        weights = 1. / torch.clamp(
            data.shortest_path_mx.view(data.num_nodes, data.num_nodes),
            min=1.0)
        cur_dist_mx_loss = (torch.abs(
            pair_mx_pred - pair_mx_true) * weights).mean()

        losses_dist_mx.append(cur_dist_mx_loss)

    loss_dist_mx = torch.stack(losses_dist_mx).mean()
    loss_cos = torch.stack(losses_cos).mean()

    return loss_dist_mx, loss_cos