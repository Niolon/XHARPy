import sys
import argparse
from .xharpy import har, create_construction_instructions
from .io import cif2data, lst2constraint_dict, write_cif, write_fcf, write_res, shelxl_hkl_to_pd


def input_with_default(question, default):
    print(question + f' [{default}]: ')
    answer = input()
    if answer.strip() == '':
        return default
    elif answer.strip() == 'exit':
        sys.exit()
    else:
        return answer.strip()



def cli(**kwargs):
    if 'cif_name' not in kwargs:
        cif_name = input_with_default(
            'Give the input .cif file',
            'iam.cif'
        )
    else:
        cif_name = kwargs['cif_name']

    if 'cif_index' not in kwargs:
        cif_index = input_with_default(
            'Give the name of the dataset in the cif file or an integer to take the nth dataset (starting with 0)',
            '0'
        )
    else:
        cif_index = kwargs['cif_index']
    if '.' not in cif_index and cif_index.isdecimal():
        cif_index = int(cif_index)

    atom_table, cell, cell_std, symm_mats_vecs, symm_strings, wavelength = cif2data(cif_name, cif_index)

    if 'lst_name' not in kwargs:
        lst_name = input_with_default(
            'Give the name of the shelxl .lst file for generation of symmetry constaints. Use "None" for no constraints',
            'iam.lst'
        )
    else:
        lst_name = kwargs['lst_name']
    if lst_name == 'None':
        constraint_dict = {}
    else:
        constraint_dict = lst2constraint_dict(lst_name)

    if 'hkl_name' not in kwargs:
        hkl_name = input_with_default(
            'Give the name of the shelxl style .hkl file (h k l intensity esd)',
            'iam.hkl'
        )
    else:
        hkl_name = kwargs['hkl_name']
    hkl = shelxl_hkl_to_pd(hkl_name)

    options_dict = {
        'xc': 'SCAN',
        'txt': 'gpaw.txt',
        'mode': 'fd',
        'h': 0.200,
        'gridinterpolation': 4,
        'average_symmequiv': False,
        'convergence':{'density': 1e-7},
        'kpts': {'size': (1, 1, 1), 'gamma': True},
        'symmetry': {'symmorphic': False}
    }
    if 'xc' not in kwargs:
        options_dict['xc'] = input_with_default(
            'Give the name of the functional for the calculation in GPAW',
            'SCAN'
        )
    else:
        options_dict['xc'] = kwargs['xc']
    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a Hirshfeld atom refinement with xHARPy from Command Line')
    parser.add_argument('--cif_name', nargs=1, help='Name of input cif file. Is assumed to be generated with shelxl')
    parser.add_argument('--cif_index', nargs=1, help='Name or index of dataset in cif file (indexing starts with 0)')
    parser.add_argument('--lst_name', nargs=1, help='Name of shelxl .lst file for generation of symmetry constaints. Use "None" for no constraints')
    parser.add_argument('--hkl_name', nargs=1, help='Name of the shelx style hkl file')
    parser.add_argument('--xc', nargs=1, help='Functional to be used by GPAW')
    parser.add_argument('--gridspacing')

    args = parser.parse_args()
    kwargs = {key: val for key, val in vars(args).items() if val is not None}
    cli(**kwargs)
