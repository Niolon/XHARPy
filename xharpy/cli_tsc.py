"""This module contains the command line interface of XHARPy. The cli function
should always accept all command line options as kwargs"""


import sys
import os
import argparse
from typing import Any
from .refine import refine
from .structure.initialise import create_construction_instructions
from .io import (cif2tsc)


def input_with_default(question: str, default: Any, auto_default: bool) -> Any:
    """Ask for a parameter value with a given default value"""
    if auto_default:
        return default
    print(question + f' [{default}]: ')
    answer = input()
    if answer.strip() == '':
        return default
    elif answer.strip() == 'exit':
        sys.exit()
    else:
        return answer.strip()


def cli(**kwargs):
    """Command line interface function. kwargs should only come from the 
    dictionary created by argparse"""
    auto_default = kwargs['auto_default']
    if 'cif_name' not in kwargs:
        cif_name = input_with_default(
            'Give the input .cif file',
            'iam.cif',
            auto_default
        )
    else:
        cif_name = kwargs['cif_name'][0]

    if 'cif_index' not in kwargs:
        cif_index = input_with_default(
            'Give the name of the dataset in the cif file or an integer to take the nth dataset (starting with 0)',
            '0',
            auto_default
        )
    else:
        cif_index = kwargs['cif_index'][0]
    if '.' not in cif_index and cif_index.isdecimal():
        cif_index = int(cif_index)

    computation_dict = {
        'txt': 'gpaw.txt',
        'mode': 'fd',
        'gridinterpolation': 4,
        'symm_equiv': 'once',
        'convergence':{'density': 1e-7},
        'symmetry': {'symmorphic': False}
    }
    if 'xc' not in kwargs:
        computation_dict['xc'] = input_with_default(
            'Give the name of the functional for the calculation in GPAW',
            'SCAN',
            auto_default
        )
    else:
        computation_dict['xc'] = kwargs['xc'][0]
    
    if 'gridspacing' not in kwargs:
        computation_dict['h'] = float(input_with_default(
            'Give the grid-spacing for the FD wavefunction calculation',
            '0.16',
            auto_default
        ))
    else:
        computation_dict['h'] = kwargs['gridspacing'][0]

    if 'kpoints' not in kwargs:
        kpoints_str = input_with_default(
            'Give the number of k-points for each direction as space separated integers',
            '1 1 1',
            auto_default
        )
        try:
            kpoints = [int(pt) for pt in kpoints_str.strip().split()]
        except ValueError as e:
            print('Could not read input. The program is expecting three integer values (e.g. "2 3 2")\n\n')
            raise e
    else:
        kpoints = kwargs['kpoints']

    computation_dict['kpts'] = {'size': tuple(kpoints), 'gamma': True}

    export_dict = {
        'core': 'constant',
    }

    if 'resolution' not in kwargs:
        export_dict['resolution_limit'] = input_with_default(
            'Give the maximum resolution in Angstrom up to which the tsc file should be generated',
            0.45,
            auto_default
        )
    else:
        export_dict['resolution_limit'] = float(kwargs['resolution'][0])

    if 'mpi_cores' not in kwargs:
        mpi_string = input_with_default(
            'Give the number of cores used for the MPI calculation (1 for no MPI, auto for letting GPAW select)',
            'auto',
            auto_default
        )
    else:
        mpi_string = kwargs['mpi_cores'][0]
    if mpi_string.strip() == 'auto':
        export_dict['f0j_source'] = 'gpaw_mpi'
    elif mpi_string.strip() == '1':
        export_dict['f0j_source'] = 'gpaw'
    else:
        computation_dict['mpicores'] = int(mpi_string)
        export_dict['f0j_source'] = 'gpaw_mpi'

    if 'output_folder' not in kwargs:
        output_folder = input_with_default(
            'Give the output folder for the fcf and the cif file',
            'xharpy_output',
            auto_default
        )
    else:
        output_folder = kwargs['output_folder'][0]

    if 'tsc_name' not in kwargs:
        tsc_name = input_with_default(
            'Give the output .tsc file',
            'xharpy.tsc',
            auto_default
        )
    else:
        tsc_name = kwargs['tsc_name'][0]

    computation_dict['save_file'] = os.path.join(output_folder, 'gpaw_density.gpw')
    computation_dict['txt'] = os.path.join(output_folder, 'gpaw.txt')



    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    cif2tsc(
        tsc_name, 
        cif_name, 
        cif_index, 
        export_dict, 
        computation_dict
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a Hirshfeld atom refinement with XHARPy from Command Line')
    parser.add_argument('--cif_name', nargs=1, help='Name of input cif file. Is assumed to be generated with shelxl')
    parser.add_argument('--cif_index', nargs=1, help='Name or index of dataset in cif file (indexing starts with 0)')
    parser.add_argument('--resolution', nargs=1, help='Resolution limit in Angstroms up to which the tsc should be generated')
    parser.add_argument('--xc', nargs=1, help='Functional to be used by GPAW')
    parser.add_argument('--gridspacing', nargs=1, type=float, help='Grid spacing for the calculation of the wavefunction')
    parser.add_argument('--kpoints', nargs=3, type=int, help='Number of k_points for the periodic calculation (3 arguments)')
    parser.add_argument('--mpi_cores', nargs=1, help='Number of cores for the multi-core calculation with mpi (1 = no mpi, auto=auto)')
    parser.add_argument('--output_folder', nargs=1, help='Folder for output of the GPAW files of the calculation')
    parser.add_argument('--tsc_name', nargs=1, help='Filename for the output tsc file')
    parser.add_argument('--auto_default', default=False, action='store_true', help='If option not present, choose the default instead of requiring a user input')


    args = parser.parse_args()
    kwargs = {key: val for key, val in vars(args).items() if val is not None}
    cli(**kwargs)
