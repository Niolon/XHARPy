import sys
import os
import argparse
from .xharpy import har, create_construction_instructions
from .io import cif2data, lst2constraint_dict, write_cif, write_fcf, shelxl_hkl_to_pd


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

    atom_table, cell, cell_std, symm_mats_vecs, symm_strings, wavelength = cif2data(cif_name, cif_index)

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
    hkl = shelxl_hkl_to_pd(hkl_name)

    if 'extinction' not in kwargs:
        extinction = input_with_default(
            'Give the type of extinction correction (none/shelxl/secondary)',
            'none'
        )
        assert extinction.lower() in ('none', 'secondary', 'shelxl'), 'Extinction can only be one of "none", "secondary" or "shelxl".'
    else:
        extinction = kwargs['extinction'][0]

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
        options_dict['xc'] = kwargs['xc'][0]
    
    if 'gridspacing' not in kwargs:
        options_dict['h'] = float(input_with_default(
            'Give the grid-spacing for the FD wavefunction calculation',
            '0.20'
        ))
    else:
        options_dict['h'] = kwargs['gridspacing'][0]

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

    options_dict['kpts'] = {'size': tuple(kpoints), 'gamma': True}

    if 'mpi_cores' not in kwargs:
        mpi_string = input_with_default(
            'Give the number of cores used for the MPI calculation (1 for no MPI, auto for letting GPAW select, experimental!)',
            '1'
        )
    else:
        mpi_string = kwargs['mpi_cores'][0]
    if mpi_string.strip() == 'auto':
        calc_source = 'gpaw_mpi'
    elif mpi_string.strip() == '1':
        calc_source = 'gpaw'
    else:
        options_dict['mpicores'] = int(mpi_string)
        calc_source = 'gpaw_mpi'

    if 'output_folder' not in kwargs:
        output_folder = input_with_default(
            'Give the output folder for the fcf and the cif file',
            'xharpy_output'
        )
    else:
        output_folder = kwargs['output_folder'][0]

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    refinement_dict = {'core': 'constant', 'extinction': extinction, 'wavelength': wavelength}
    
    construction_instructions, parameters = create_construction_instructions(
        atom_table,
        constraint_dict,
        refinement_dict=refinement_dict
    )

    parameters, var_cov_mat, information = har(
        cell,
        symm_mats_vecs,
        hkl,
        construction_instructions,
        parameters,
        options_dict=options_dict,
        f0j_source=calc_source,
        reload_step=1,
        refinement_dict=refinement_dict
    )

    write_cif(
        os.path.join(output_folder, 'xharpy.cif'),
        'xharpy_refinement',
        shelx_cif_name=cif_name,
        shelx_descr=cif_index,
        source_cif_name=cif_name,
        source_descr=cif_index,
        fjs=information['fjs_anom'],
        parameters=parameters,
        var_cov_mat=var_cov_mat,
        construction_instructions=construction_instructions,
        symm_mats_vecs=symm_mats_vecs,
        hkl=hkl,
        shift_ov_su=information['shift_ov_su'],
        options_dict=options_dict,
        refine_dict=refinement_dict,
        cell=cell,
        cell_std=cell_std
    )

    write_fcf(
        filename=os.path.join(output_folder, 'xharpy.fcf'),
        hkl=hkl,
        refine_dict=refinement_dict,
        parameters=parameters,
        symm_strings=symm_strings,
        construction_instructions=construction_instructions,
        fjs=information['fjs_anom'],
        cell=cell,
        dataset_name='xharpy_refinement',
        fcf_mode=4
    )

    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a Hirshfeld atom refinement with xHARPy from Command Line')
    parser.add_argument('--cif_name', nargs=1, help='Name of input cif file. Is assumed to be generated with shelxl')
    parser.add_argument('--cif_index', nargs=1, help='Name or index of dataset in cif file (indexing starts with 0)')
    parser.add_argument('--lst_name', nargs=1, help='Name of shelxl .lst file for generation of symmetry constaints. Use "None" for no constraints')
    parser.add_argument('--hkl_name', nargs=1, help='Name of the shelx style hkl file')
    parser.add_argument('--extinction', nargs=1, help='Type of extinction refinement must be either none/shelxl/secondary', choices=['none', 'shelxl', 'secondary'])
    parser.add_argument('--xc', nargs=1, help='Functional to be used by GPAW')
    parser.add_argument('--gridspacing', nargs=1, type=float, help='Grid spacing for the calculation of the wavefundction')
    parser.add_argument('--kpoints', nargs=3, type=int, help='Number of k_points for the periodic calculation (3 arguments)')
    parser.add_argument('--mpi_cores', nargs=1, help='Number of cores for the multi-core calculation with mpi (1 = no mpi, auto=auto)')
    parser.add_argument('--output_folder', nargs=1, help='Folder for output of cif and fcf file')

    args = parser.parse_args()
    kwargs = {key: val for key, val in vars(args).items() if val is not None}
    cli(**kwargs)
