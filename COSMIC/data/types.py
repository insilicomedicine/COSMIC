atom_to_idx = {'start': 0,
               'C': 1,
               'N': 2,
               'O': 3,
               'S': 4,
               'F': 5,
               'Cl': 6,
               'Br': 7,
               'I': 8,
               'P': 9,
               'H': 10,
               'Mn': 11,
               'Mg': 12,
               'Au': 13,
               'Ge': 14,
               'Pt': 15,
               'Ag': 16,
               'As': 17,
               'Se': 18,
               'Ga': 19,
               'Na': 20,
               'K': 21,
               'Bi': 22,
               'Cr': 23,
               'Li': 24,
               'V': 25,
               'B': 26,
               'Si': 27,
               'Ca': 28,
               'Zn': 29,
               'Gd': 30,
               'In': 31,
               'Sb': 32,
               'Al': 33,
               'Hg': 34,
               'Cu': 35,
               'unknown': 36}

bond_to_idx = {
    'None': 0,
    'start': 1,
    'SINGLE': 2,
    'DOUBLE': 3,
    'TRIPLE': 4,
    'AROMATIC': 5,
    'LEVEL': 6,
    'SECOND_ORDER': 7,
    'THIRD_ORDER': 8}

stereo_to_idx = {
    'None': 0,
    ('forward', 'STEREONONE'): 1,
    ('forward', 'STEREOANY'): 2,
    ('forward', 'STEREOZ'): 3,
    ('forward', 'STEREOE'): 4,
    ('forward', 'STEREOCIS'): 5,
    ('forward', 'STEREOTRANS'): 6,
    ('backward', 'STEREONONE'): 7,
    ('backward', 'STEREOANY'): 8,
    ('backward', 'STEREOZ'): 9,
    ('backward', 'STEREOE'): 10,
    ('backward', 'STEREOCIS'): 11,
    ('backward', 'STEREOTRANS'): 12
}

chiral_to_idx = {
    'None': 0,
    'CHI_UNSPECIFIED': 1,
    'CHI_TETRAHEDRAL_CW': 2,
    'CHI_TETRAHEDRAL_CCW': 3,
    'CHI_OTHER': 4}

charge_to_idx = {
    'None': 0,
    'start': 1,
    -3: 2,
    -2: 3,
    -1: 4,
    0: 5,
    1: 6,
    2: 7,
    3: 8}


class NoStartTriplePoints(Exception):
    pass


class CollinearPointsException(Exception):
    pass
