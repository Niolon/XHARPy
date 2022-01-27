from typing import List, Tuple, Dict, Any

from ..conversion import cell_constants_to_M
from ..core import construct_values, AtomInstructions
from ..io import symm_to_matrix_vector
from .tsc_file_source import calc_f0j as calc_f0j_tsc

import numpy as np
import fractions
import subprocess
import pandas as pd
import re
import os
import time


def generate_cube_shell(shell_index: int) -> np.ndarray:
    """Generates indicees for the surface of a cube that is constructed from
    (2 * shell_index + 1)**3 smaller indexed cubes, e.g. if shell_index
    is 2 will generate all combinations of three coordinates where one 
    coordinate is 2 or -2. Is used for building larger and larger shells of
    surrounding unit cells.

    Parameters
    ----------
    shell_index : int
        index of cube shell, index 0 is only one center cube, index one
        adds the 26 surronding smaller cubes and so on

    Returns
    -------
    shell_add: np.ndarray
        size (s,3) array of coordinates of surrounding unit cells
    """    
    
    if shell_index == 0:
        shell = np.array([[0,0,0]])
    else:
        # generate shell_indexes of full front in x direction
        abc = np.meshgrid(np.array([shell_index]), np.arange(-shell_index, shell_index + 1), np.arange(-shell_index, shell_index + 1))
        addx1 = np.array(abc).reshape(3, 4*shell_index**2 + 4 * shell_index + 1).T
        addx2 = addx1.copy()
        addx2[:,0] = -shell_index

        # in y direction do generate a rectangle to not count edge shell_indexes multiple times
        abc = np.meshgrid(np.arange(-shell_index + 1, shell_index), np.array([shell_index]), np.arange(-shell_index, shell_index + 1))
        addy1 = np.array(abc).reshape(3, 4*shell_index**2 - 1).T
        addy2 = addy1.copy()
        addy2[:,1] = -shell_index

        # in z only generate the left over square in the center
        abc = np.meshgrid(np.arange(-shell_index + 1, shell_index), np.arange(-shell_index + 1, shell_index),  np.array([shell_index]))
        addz1 = np.array(abc).reshape(3, 4*shell_index**2 -4*shell_index + 1).T
        addz2 = addz1.copy()
        addz2[:,2] = -shell_index
        shell = np.concatenate((addx1, addx2, addy1, addy2, addz1, addz2))
    return shell

def generate_cluster_file(
    symm_mats_vecs: Tuple[np.ndarray, np.ndarray],
    fragment_positions: np.ndarray,
    original_labels: List[str],
    cell_mat_m: np.ndarray,
    cutoff: float,
    calc_folder: str,
    charge: float
) -> str:
    """Will read the Hirshfeld charges from a NoSpherA2.log file and then 
    generate symmetry equivalent positions, where at least one atom from the 
    fragment is within the cutoff (in Angstrom). The charges are written
    into a 'cluster_charges.pc' file and the string to add to the orca.inp
    is returned. May be inefficient so if you run out of memory try reducing
    the cluster radius

    Parameters
    ----------
    symm_mats_vecs : Tuple[np.ndarray, np.ndarray]
        size (K, 3, 3) array of symmetry  matrices and (K, 3) array of
        translation vectors for all symmetry elements in the unit cell
    fragment_positions : np.ndarray
        size (Z, 3) array of positions of the fragment (including added
        atoms from a build_dict)
    original_labels : List[str]
        original_labels of the fragment positions, used for lookup from the 
        NoSpherA2.log
    cell_mat_m : np.ndarray
        size (3, 3) array with the unit cell vectors as row vectors
    cutoff : float
        cutoff radius in Angstrom
    calc_folder : str
        calc_folder for the NoSpherA2 and ORCA calculation
    charge : float
        Overall charge. The sum of atom charges from the fragment will
        be corrected to be this value, to correct rounding errors.

    Returns
    -------
    charge_str : str
        Will be an empty string if no cluster charges were added, otherwise
        will return the string, that needs to be added to the ORCA input file.
    """
    if abs(cutoff) < 1e-10:
        return ''
    symm_mats, symm_vecs = symm_mats_vecs
    symm_mats = np.array(symm_mats)
    symm_vecs = np.array(symm_vecs)
    #einsum shell_indexes: k: n_symm, z: n_atom, s:shell
    symm_positions = np.einsum('kxy, zy -> kzx', symm_mats, fragment_positions) + symm_vecs[:, None, :]
    cart = np.einsum('xy, zy -> zx', cell_mat_m, fragment_positions)

    out_of_range = False

    new_positions = np.empty(shape=(0,3))
    new_indexes = np.empty(shape=(0), dtype=np.int64)
    for shell_index in range(100):
        shell = generate_cube_shell(shell_index)
        symm_shell_pos = shell[:, None, None, :] + symm_positions[None, :, :, :]
        symm_shell_cart = np.einsum('xy, skzx -> skzy', cell_mat_m, symm_shell_pos)
        distances = np.linalg.norm(symm_shell_cart[:,:,None, :, :,] - cart[None, None, :, None, :], axis=-1)
        distance_criterion = distances < cutoff
        non_self = distances > 1e-6
        select = np.logical_and(np.any(distance_criterion, axis=(-1,-2))[:,:,None], np.all(non_self, axis=-1))
        if np.sum(select) == 0:
            if out_of_range:
                # coordinates might be outside unit cell, so for safety we wait for two empty shells
                break
            out_of_range = True
        else:
            out_of_range = False
            new_positions= np.append(new_positions, symm_shell_cart[select], axis=0)
            new_indexes = np.append(new_indexes, np.where(select)[2], axis=0)

    _, unique_indexes = np.unique(np.round(new_positions, 5), axis=0, return_index=True)

    cluster_positions = new_positions[unique_indexes]


    with open(os.path.join(calc_folder, 'NoSpherA2.log'), 'r') as fo:
        content = fo.read()

    charge_tab_match = re.search(r'Atom\s+Becke\s+Spherical\s+Hirshfeld(.*)\nTotal number of electrons', content, flags=re.DOTALL)

    assert charge_tab_match is not None, 'Could not find charge table in NoSpherA2.log, probably unexpected format'

    charge_tab = charge_tab_match.group(1)
    charge_dict = {}
    for line in charge_tab.split('\n')[1:]:
        name, _, _, atom_charge = line.strip().split()
        charge_dict[name] = float(atom_charge)
    fragment_charges = np.array([charge_dict[label] for label in original_labels])
    
    fragment_charges -= (np.sum(fragment_charges) - charge) / len(fragment_charges)

    cluster_charges = fragment_charges[new_indexes[unique_indexes]]
    print(f'  constructed cluster, the overall charge is: {np.sum(cluster_charges)}')

    #strings = []
    #for charge, position in zip(cluster_charges, cluster_positions):
    #    strings.append(f'Q {charge: 7.4f} {position[0]: 15.10f} {position[1]: 15.10f} {position[2]: 15.10f}')
        
    strings = []
    for charge, position in zip(cluster_charges, cluster_positions):
        strings.append(f'{charge: 7.4f} {position[0]: 15.10f} {position[1]: 15.10f} {position[2]: 15.10f}')

    with open(os.path.join(calc_folder, 'cluster_charges.pc'), 'w') as fo:
        fo.write(f'{len(strings)}\n')
        fo.write('\n'.join(strings))
        fo.write('\n\n')
    return '%pointcharges "cluster_charges.pc"'


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors v1 and v2"""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def calc_f0j(
    cell_mat_m: np.ndarray,
    construction_instructions: List[AtomInstructions],
    parameters: np.ndarray,
    index_vec_h: np.ndarray,
    symm_mats_vecs: Tuple[np.ndarray, np.ndarray],
    computation_dict: Dict[str, Any],
    restart: bool = True,
    explicit_core: bool = True
)-> np.ndarray:
    """Gives the possibility to read atomic form factors from a .tsc file.
    For the format see: https://arxiv.org/pdf/1911.08847.pdf. Has only been 
    tested with SYMM:expanded options and does not tolerate duplicate hkl
    indicees.
    Is also meant for fast prototyping (see more about that in the
    'call_function' keyword in the computation_dict).

    Parameters
    ----------
    cell_mat_m : np.ndarray
        size (3, 3) array with the unit cell vectors as row vectors
    construction_instructions : List[AtomInstructions]
        List of instructions for reconstructing the atomic parameters from the
        list of refined parameters
    parameters : np.ndarray
        Current parameter values
    index_vec_h : np.ndarray
        size (H) vector containing Miller indicees of the measured reflections
    symm_mats_vecs : Tuple[np.ndarray, np.ndarray]
        size (K, 3, 3) array of symmetry  matrices and (K, 3) array of
        translation vectors for all symmetry elements in the unit cell
    computation_dict : Dict[str, Any]
        Contains options for NoSpherA2/ORCA calculation
         - orca_path (str): Path to the ORCA executable. Is required for 
           multi-core calculations. If the path given is relative, take into
           account that the executable will be run in the calc_folder and 
           change the relative path accordingly, by default 'orca'
         - nosphera2_path (str): Path to the NoSpherA2 executable. Needs to be
           given, as with the orca_path, a relative path needs to be given
           relative to the calc_folder, not the folder, where the script is run.
         - nosphera2_accuracy (int): Number between 1-5 for the size of the 
           grid nosphera2 uses for the calculation of atomic form factors,
           by default 3
         - calc_folder (str): Folder in which the ORCA and NoSpherA2 
           calculations will be conducted. Mainly used to keep the main 
           directory somewhat clean, by default 'calc'
         - basis_set (str): if there is no newline character this string will
           be used by ORCA to selet the basis set by name (e.g. def2-TZVPP).
           If a newline is present, it will instead be used within ORCA's 
           %basis keyword, with a single 'end' added at the end. This way
           basis sets from EMSL can be used, by defailt 'def2-SVP'
         - functional (str): Density functional as ORCA keyword, by default
           'PBE'
         - charge (float): Fragment charge, by default 0
         - multiplicity (int): Fragment multiplicity, by default 1
         - n_cores (int): number of cores used for the ORCA and NoSpherA2
           calculations. If larger than 1, the orca path needs to be given
           as an absolute path, by default 1.
         - cutoff (float): Cutoff in Angstrom for the generation of cluster-
           charges. Fragments, which have an atom within this radius will
           be added completely to the cluster charge list, by default 0.0
         - build_dict (Dict[str, List(str)]): Dictionary to complete a 
           fragment before the calculation. The key string needs to be a 
           symmetry card (e.g. -x, 1-y, 1/2-z). The following list needs to
           contain the atom names, on which the symmetry is supposed to be 
           applied.

    restart : bool, optional
        If true, the DFT calculation will be restarted from a previous calculation
    explicit_core : bool, optional
        If True the frozen core density is assumed to be calculated separately, 
        therefore only the valence density will be split up, by default True

    Returns
    -------
    f0j : np.ndarray
        size (K, N, H) array of atomic form factors for all reflections and symmetry
        generated atoms within the unit cells. Atoms on special positions are 
        present multiple times and have the atomic form factor of the full atom.
    """
    labels = [instr.name for instr in construction_instructions]

    element_symbols = [instr.element for instr in construction_instructions]

    positions, *_ = construct_values(
        parameters,
        construction_instructions,
        cell_mat_m
    )
    positions = np.array(positions)
    
    build_dict = computation_dict.get('build_dict', {})
    calc_folder = computation_dict.get('calc_folder', 'calc')
    cutoff = computation_dict.get('cutoff', 0.0)
    basis_set = computation_dict.get('basis_set', 'def2-SVP')
    functional = computation_dict.get('functional', 'PBE')
    charge = computation_dict.get('charge', 0)
    multiplicity = computation_dict.get('multiplicity', 1)
    n_cores = computation_dict.get('n_cores', 1)
    orca_path = computation_dict.get('orca_path', 'orca')
    assert orca_path != '.' or n_cores == 1, 'for multicore calculations you need to give an absolute orca_path'
    assert 'nosphera2_path' in computation_dict, 'Give a path to the NoSpherA2 Executable'
    nosphera2_path = computation_dict['nosphera2_path']
    nosphera2_accuracy = computation_dict.get('nosphera2_accuracy', 3)

    fragment_positions = positions.copy()
    fragment_elements = element_symbols.copy()
    fragment_labels = labels.copy()
    original_labels = labels.copy()
    for symm_index, (key, val) in enumerate(build_dict.items()):
        mat, vec = symm_to_matrix_vector(key)
        indexes = tuple(labels.index(name) for name in val)
        new_positions = np.einsum('xy, zy -> zx', mat, positions[indexes,:]) + vec
        new_elements = [element_symbols[index] for index in indexes]
        fragment_positions = np.concatenate((fragment_positions, new_positions))
        fragment_elements += new_elements
        new_labels = [labels[index] + str(symm_index + 1) for index in indexes]
        fragment_labels += new_labels
        new_labels = [labels[index] for index in indexes]
        original_labels += new_labels

    if not os.path.exists(calc_folder):
        os.mkdir(calc_folder)

    cart_positions = np.einsum('xy, zy -> zx', cell_mat_m, fragment_positions)
    
    # create cluster charge file
    if restart and abs(cutoff) > 1e-10:    
        charge_str = generate_cluster_file(
            symm_mats_vecs,
            fragment_positions, 
            original_labels, 
            cell_mat_m, 
            cutoff, 
            calc_folder,
            charge
    )
    else:
        charge_str = ''

    # create orca file
    xyz_str = [f'{elem:<2s}{pos[0]:15.10f}{pos[1]:15.10f}{pos[2]:15.10f}'
               for elem, pos in zip(fragment_elements, cart_positions)]
        
    if not restart:
        autostart = 'NoAutostart'
    else:
        autostart = ''

    if '\n' in basis_set:
        multiline_basis = '%basis\n' + basis_set + '\nend'
        basis_set = '3-21G'
    else:
        multiline_basis = ''

    lines = [
        f'! NoPop MiniPrint {basis_set} AIM {functional} DefGrid3 NoFinalGridX VeryTightSCF NormalConv {autostart}',
        f'%pal nprocs {n_cores} end',
        charge_str,
        '',
        '%coords',
        '    CTyp xyz',
        f'    charge {charge}',
        f'    mult {multiplicity}',
        '    units angs',
        '    coords',
        *xyz_str,
        '    end',
        'end',
        '',
        multiline_basis,
        ''
    ]

    with open(os.path.join(calc_folder, 'orca.inp'), 'w') as fo:
        fo.write('\n'.join(lines))

    # create hkl file
    sort_df = pd.DataFrame({'h': index_vec_h[:,0], 'k': index_vec_h[:,1],  'l': index_vec_h[:,2]})

    with open(os.path.join(calc_folder, 'mock.hkl'), 'w') as fo:
        for _, row in sort_df.sort_values(['h', 'k', 'l']).iterrows():
            fo.write(f"{int(row['h']):4d}{int(row['k']):4d}{int(row['l']):4d}{1.0:8.2f}{0.1:8.2f}\n")


    # create cif file
    # create symmetry cards in format that NoSpherA2 expects
    symm_string = ''
    for index, (symm_mat, symm_vec) in enumerate(zip(*symm_mats_vecs)):
        symm_string += f" {index + 1} '"
        for symm_parts, add in zip(symm_mat, symm_vec):
            symm_string_add = str(fractions.Fraction(add).limit_denominator(50))
            if symm_string_add != '0':
                symm_string += symm_string_add 
            for symm_part, symbol in zip(symm_parts, ['X', 'Y', 'Z']):
                if abs(symm_part) < 1e-10:
                    continue
                if abs(1 - abs(symm_part)) < 1e-10:
                    if symm_part > 0:
                        symm_string += f'+{symbol}'
                    else:
                        symm_string += f'-{symbol}'
                else:
                    fraction = fractions.Fraction(symm_part).limit_denominator(50)
                    if str(fraction).startswith('-'):
                        symm_string += f'{str(fraction)}*{symbol}'
                    else:
                        symm_string += f'+{str(fraction)}*{symbol}'
            symm_string += ','
        symm_string = symm_string[:-1] + "'\n"

    a, b, c = np.linalg.norm(cell_mat_m, axis=0)
    alpha = np.rad2deg(angle_between(cell_mat_m[:,1], cell_mat_m[:,2]))
    beta = np.rad2deg(angle_between(cell_mat_m[:,0], cell_mat_m[:,2]))
    gamma = np.rad2deg(angle_between(cell_mat_m[:,0], cell_mat_m[:,1]))
    v = np.linalg.det(cell_mat_m)

    position_strings = []
    for label, element, xyz in zip(labels, element_symbols, positions):
        position_strings.append(f' {label} {element} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f} . .')
    position_output = '\n'.join(position_strings)

    fragment_strings = []
    for label, element, xyz in zip(fragment_labels, fragment_elements, fragment_positions):
        fragment_strings.append(f' {label} {element} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f} . .')
    fragment_output = '\n'.join(fragment_strings)

    cif_string = f"""
data_2nosphera2

_cell_length_a             {a:<12.8f}
_cell_length_b             {b:<12.8f}
_cell_length_c             {c:<12.8f}
_cell_angle_alpha          {alpha:<12.8f}
_cell_angle_beta           {beta:<12.8f}
_cell_angle_gamma          {gamma:<12.8f}
_cell_volume               {v:<12.8f}

loop_
 _space_group_symop_id
 _space_group_symop_operation_xyz
{symm_string}
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_disorder_group
 _atom_site_disorder_assembly
"""

    with open(os.path.join(calc_folder, 'npa2_asym.cif'), 'w') as fo:
        fo.write(cif_string + position_output + '\n\n')
        
    with open(os.path.join(calc_folder, 'npa2.cif'), 'w') as fo:
        fo.write(cif_string + fragment_output + '\n\n')

    subprocess.check_call([f'{orca_path} orca.inp > orca_log.log'], shell=True, cwd=calc_folder)
    time.sleep(1)
    subprocess.check_call(f'{nosphera2_path} -hkl mock.hkl -wfn orca.wfn -cif npa2.cif -asym_cif npa2_asym.cif -multiplicity {multiplicity} -acc {nosphera2_accuracy} -cores {n_cores}', shell=True, stdout=subprocess.DEVNULL, cwd=calc_folder)

    tsc_dict = {
        'file_name': os.path.join(calc_folder, 'experimental.tsc')
    }

    return calc_f0j_tsc(
        cell_mat_m,
        construction_instructions,
        parameters,
        index_vec_h,
        symm_mats_vecs,
        tsc_dict,
        restart,
        explicit_core
    )

def calc_f0j_core(
    *args, **kwargs
):
    raise ValueError('Separate core calculation is currently not available for ORCA-NoSpherA2 calculations')


def generate_cif_output(
    computation_dict: Dict[str, Any]
) -> str:
    """Generates at string, that details the computation options for use in the 
    cif generation routine.

    Parameters
    ----------
    computation_dict : Dict[str, Any]
        contains options for the calculation.

    Returns
    -------
    str
        The string that will be added to the cif-file
    """

    strings = []
    for key, val in computation_dict.items():
        if type(val) is dict:
            strings.append(f'      {key}:')
            for key2, val2 in val.items():
                strings.append(f'         {key2}: {val2}')
        else:
            strings.append(f'      {key}: {val}')
    value_strings = '\n'.join(strings)
    addition = f"""  - Refinement was done using structure factors
    derived from theoretically calculated densities
  - Density calculation was done with ORCA using the
    following settings
{value_strings}
  - Afterwards density was partitioned according to the Hirshfeld scheme,
    using NoSpherA2, which was also used to calculate atomic form factors"""
    return addition