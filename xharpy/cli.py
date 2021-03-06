"""This module contains the command line interface of XHARPy. The cli function
should always accept all command line options as kwargs"""


import sys
import os
import argparse
from typing import Any
from .refine import refine
from .structure.initialise import create_construction_instructions
from .io import (cif2data, lst2constraint_dict, write_cif, write_fcf, 
                 shelxl_hkl2pd, write_res)


def input_with_default(question: str, default: Any) -> Any:
    """Ask for a parameter value with a given default value"""
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
    if 'cif_name' not in kwargs:
        cif_name = input_with_default(
            'Give the input .cif file',
            'iam.cif'
        )
    else:
        cif_name = kwargs['cif_name'][0]

    if 'cif_index' not in kwargs:
        cif_index = input_with_default(
            'Give the name of the dataset in the cif file or an integer to take the nth dataset (starting with 0)',
            '0'
        )
    else:
        cif_index = kwargs['cif_index'][0]
    if '.' not in cif_index and cif_index.isdecimal():
        cif_index = int(cif_index)

    atom_table, cell, cell_esd, symm_mats_vecs, symm_strings, wavelength = cif2data(cif_name, cif_index)

    if 'lst_name' not in kwargs:
        lst_name = input_with_default(
            'Give the name of the shelxl .lst file for generation of symmetry constaints. Use "None" for no constraints',
            'iam.lst'
        )
    else:
        lst_name = kwargs['lst_name'][0]
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
        hkl_name = kwargs['hkl_name'][0]
    hkl = shelxl_hkl2pd(hkl_name)

    if 'extinction' not in kwargs:
        extinction = input_with_default(
            'Give the type of extinction correction (none/shelxl/secondary)',
            'none'
        )
        assert extinction.lower() in ('none', 'secondary', 'shelxl'), 'Extinction can only be one of "none", "secondary" or "shelxl".'
    else:
        extinction = kwargs['extinction'][0]

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
            'SCAN'
        )
    else:
        computation_dict['xc'] = kwargs['xc'][0]
    
    if 'gridspacing' not in kwargs:
        computation_dict['h'] = float(input_with_default(
            'Give the grid-spacing for the FD wavefunction calculation',
            '0.16'
        ))
    else:
        computation_dict['h'] = kwargs['gridspacing'][0]

    if 'kpoints' not in kwargs:
        kpoints_str = input_with_default(
            'Give the number of k-points for each direction as space separated integers',
            '1 1 1'
        )
        try:
            kpoints = [int(pt) for pt in kpoints_str.strip().split()]
        except ValueError as e:
            print('Could not read input. The program is expecting three integer values (e.g. "2 3 2")\n\n')
            raise e
    else:
        kpoints = kwargs['kpoints']

    computation_dict['kpts'] = {'size': tuple(kpoints), 'gamma': True}

    refinement_dict = {
        'core': 'constant',
        'extinction': extinction
    }

    if 'mpi_cores' not in kwargs:
        mpi_string = input_with_default(
            'Give the number of cores used for the MPI calculation (1 for no MPI, auto for letting GPAW select, experimental!)',
            '1'
        )
    else:
        mpi_string = kwargs['mpi_cores'][0]
    if mpi_string.strip() == 'auto':
        refinement_dict['f0j_source'] = 'gpaw_mpi'
    elif mpi_string.strip() == '1':
        refinement_dict['f0j_source'] = 'gpaw'
    else:
        computation_dict['mpicores'] = int(mpi_string)
        refinement_dict['f0j_source'] = 'gpaw_mpi'

    if 'output_folder' not in kwargs:
        output_folder = input_with_default(
            'Give the output folder for the fcf and the cif file',
            'xharpy_output'
        )
    else:
        output_folder = kwargs['output_folder'][0]

    computation_dict['save_file'] = os.path.join(output_folder, 'gpaw_density.gpw')
    computation_dict['txt'] = os.path.join(output_folder, 'gpaw.txt')

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    construction_instructions, parameters = create_construction_instructions(
        atom_table,
        refinement_dict,
        constraint_dict
    )

    parameters, var_cov_mat, information = refine(
        cell,
        symm_mats_vecs,
        hkl,
        construction_instructions,
        parameters,
        wavelength=wavelength,
        computation_dict=computation_dict,
        refinement_dict=refinement_dict
    )

    write_cif(
        output_cif_path=os.path.join(output_folder, 'xharpy.cif'),
        cif_dataset='xharpy',
        shelx_cif_path=cif_name,
        shelx_dataset=cif_index,
        cell=cell,
        cell_esd=cell_esd,
        symm_mats_vecs=symm_mats_vecs,
        hkl=hkl,
        construction_instructions=construction_instructions,
        parameters=parameters,
        var_cov_mat=var_cov_mat,
        refinement_dict=refinement_dict,
        computation_dict=computation_dict,
        information=information
    )

    write_fcf(
        fcf_path=os.path.join(output_folder, 'xharpy.fcf'),
        fcf_dataset='xharpy',
        fcf_mode=4,
        cell=cell,
        hkl=hkl,
        construction_instructions=construction_instructions,
        parameters=parameters,
        wavelength=wavelength,
        refinement_dict=refinement_dict,
        symm_strings=symm_strings,
        information=information,
    )

    write_res(
        out_res_path=os.path.join(output_folder, 'xharpy_6.res'),
        in_res_path=lst_name,
        cell=cell,
        cell_esd=cell_esd,
        construction_instructions=construction_instructions,
        parameters=parameters,
        wavelength=wavelength
    )

    write_fcf(
        fcf_path=os.path.join(output_folder, 'xharpy_6.fcf'),
        fcf_dataset='xharpy_6',
        fcf_mode=6,
        cell=cell,
        hkl=hkl,
        construction_instructions=construction_instructions,
        parameters=parameters,
        wavelength=wavelength,
        refinement_dict=refinement_dict,
        symm_strings=symm_strings,
        information=information,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a Hirshfeld atom refinement with XHARPy from Command Line')
    parser.add_argument('--cif_name', nargs=1, help='Name of input cif file. Is assumed to be generated with shelxl')
    parser.add_argument('--cif_index', nargs=1, help='Name or index of dataset in cif file (indexing starts with 0)')
    parser.add_argument('--lst_name', nargs=1, help='Name of shelxl .lst file for generation of symmetry constaints. Use "None" for no constraints')
    parser.add_argument('--hkl_name', nargs=1, help='Name of the shelx style hkl file')
    parser.add_argument('--extinction', nargs=1, help='Type of extinction refinement must be either none/shelxl/secondary', choices=['none', 'shelxl', 'secondary'])
    parser.add_argument('--xc', nargs=1, help='Functional to be used by GPAW')
    parser.add_argument('--gridspacing', nargs=1, type=float, help='Grid spacing for the calculation of the wavefunction')
    parser.add_argument('--kpoints', nargs=3, type=int, help='Number of k_points for the periodic calculation (3 arguments)')
    parser.add_argument('--mpi_cores', nargs=1, help='Number of cores for the multi-core calculation with mpi (1 = no mpi, auto=auto)')
    parser.add_argument('--output_folder', nargs=1, help='Folder for output of cif and fcf file')

    args = parser.parse_args()
    kwargs = {key: val for key, val in vars(args).items() if val is not None}
    cli(**kwargs)
