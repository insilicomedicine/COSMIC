import os
import json

import tqdm

from collections import defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import SDWriter
from rdkit.Chem.Scaffolds import MurckoScaffold

import pickle

import argparse


def compute_scaffold(sm, min_rings=2):
    try:
        mol = Chem.MolFromSmiles(sm)
        Chem.rdmolops.RemoveStereochemistry(mol)

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return sm

    n_rings = scaffold.GetRingInfo().NumRings()
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if scaffold_smiles == '' or n_rings < min_rings:
        return sm
    return scaffold_smiles


def scaffold_split(smiles_with_scaffolds, seed=2021):
    np.random.seed(seed)
    smiles_by_scaffold = defaultdict(lambda: [])
    for sm, sc in smiles_with_scaffolds:
        smiles_by_scaffold[sc].append(sm)

    scaffolds_sorted = sorted(list(smiles_by_scaffold.keys()))
    scaffolds_permuted = np.random.permutation(scaffolds_sorted)

    train_scaffolds = scaffolds_permuted[:int(0.85 * len(scaffolds_permuted))]
    val_scaffolds = scaffolds_permuted[int(0.85 * len(scaffolds_permuted)):
                                       int(0.9 * len(scaffolds_permuted))]
    test_scaffolds = scaffolds_permuted[int(0.9 * len(scaffolds_permuted)):]

    train_smiles = []
    val_smiles = []
    test_smiles = []

    for sc in train_scaffolds:
        subsets = ['full']
        if np.random.rand() < 0.1:
            subsets.append('small')
        train_smiles += [(s, subsets) for s in smiles_by_scaffold[sc]]

    for sc in val_scaffolds:
        subsets = ['full']
        if np.random.rand() < 0.1:
            subsets.append('small')
        val_smiles += [(s, subsets) for s in smiles_by_scaffold[sc]]

    for sc in test_scaffolds:
        subsets = ['full']
        if np.random.rand() < 0.1:
            subsets.append('small')
        test_smiles += [(s, subsets) for s in smiles_by_scaffold[sc]]

    return train_smiles, val_smiles, test_smiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str)
    parser.add_argument('--summary_file', type=str)
    parser.add_argument('--new_summary_file', type=str)
    parser.add_argument('--seed', type=str, default=2021)

    args = parser.parse_args()

    summary_file = os.path.join(args.root, args.summary_file)

    with open(summary_file, "r") as f:
        summary_file = json.load(f)

    smiles = []

    for sm in tqdm.tqdm(summary_file.keys()):
        if 'pickle_path' not in summary_file[sm]:
            continue

        pickle_path = summary_file[sm]['pickle_path']
        mol_path = os.path.join(args.root, pickle_path)

        with open(mol_path, "rb") as f:
            confs = [conf for conf in pickle.load(f)['conformers']]

        if any([len(Chem.GetMolFrags(conf['rd_mol'])) != 1 for conf in confs]):
            continue

        sdf_path = os.path.join(pickle_path.split('/')[0] + '_sdf',
                                pickle_path.split('/')[1].replace('.pickle',
                                                                  '.sdf'))
        sdf_writer = SDWriter(os.path.join(args.root, sdf_path))
        sdf_writer.SetProps(['totalenergy'])

        for conf in confs:
            mol = conf['rd_mol']
            mol.SetDoubleProp('totalenergy', conf['totalenergy'])
            sdf_writer.write(mol)

        sdf_writer.flush()
        sdf_writer.close()

        summary_file[sm].pop('pickle_path')
        summary_file[sm]['sdf_path'] = sdf_path

        smiles.append(sm)

    smiles_with_scaffolds = [(sm, compute_scaffold(sm)) for sm in smiles]

    train_smiles, val_smiles, test_smiles = scaffold_split(
        smiles_with_scaffolds,
        seed=args.seed)

    for sm, subsets in train_smiles:
        summary_file[sm]['split'] = 'train'
        summary_file[sm]['subsets'] = subsets
    for sm, subsets in val_smiles:
        summary_file[sm]['split'] = 'val'
        summary_file[sm]['subsets'] = subsets
    for sm, subsets in test_smiles:
        summary_file[sm]['split'] = 'test'
        summary_file[sm]['subsets'] = subsets

    new_summary_file = os.path.join(args.root, args.new_summary_file)
    with open(new_summary_file, "w") as f:
        json.dump(summary_file, f)
