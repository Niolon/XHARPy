import numpy as np
import pandas as pd

from typing import List, Dict, Tuple, Any
from ..core import AtomInstructions, construct_values



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
        Contains options for the .tsc source
        - file_name (str): Path to the .tsc file, by default 'to_xharpy.tsc'

        - call_function (python function): If this option is not 'none' you can
          pass a function, which will be called in each Hirshfeld cycle. The
          function receives four arguments: labels contrains the atom label for 
          each atom in the asymmetric unit, element_symbols containts the 
          element symbols (e.g. H, Na), positions are the atomic positions in 
          FRACTIONAL coordinates, restart is a bool, which you can check to 
          trigger a start of a calculation from a precalculated density. 
          (Usually you would want to start the first step with a calculation 
          from scratch and then recycle for all other HAR cycles, as the
          differences in posutions get smaller). At the end of the function
          you should write a new .tsc file with the atomic form factors,
          by default 'none'

        - call_args (List): If you have a call_function you can use this
          option to pass additional arguments, which will be passed after
          the four default arguments, by default []

        - call_kwargs (Dict): If you have a call_function you can use this
          option to pass additional keyword arguments, by default {}

        - cif_addition (str): Will be added to the refinement_details section 
          of the cif_file
    
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
    
    if 'file_name' in computation_dict:
        file_name = computation_dict['file_name']
    else:
        file_name = 'to_xharpy.tsc'

    labels = [instr.name for instr in construction_instructions]

    if 'call_args' in computation_dict:
        call_args = computation_dict['call_args']
    else:
        call_args = []

    if 'call_kwargs' in computation_dict:
        call_kwargs = computation_dict['call_kwargs']
    else:
        call_kwargs = {}

    if 'call_function' in computation_dict and computation_dict['call_function'] != 'none':
        call_function = computation_dict['call_function']

        element_symbols = [instr.element for instr in construction_instructions]

        positions, *_ = construct_values(
            parameters,
            construction_instructions,
            cell_mat_m
        )

        call_function(
            labels,
            element_symbols,
            positions,
            restart,
            *call_args,
            **call_kwargs
        )

    with open(file_name, 'r') as fo:
        content = fo.read()

    header, data = content.split('DATA:')

    header_lines = header.strip().split('\n')
    scatterers = next(line for line in header_lines if line.startswith('SCATTERERS'))
    scatterers = scatterers.split(':')[1].strip().split()
    try:
        ad_line = next(line for line in header_lines if line.startswith('SCATTERERS'))
        includes_ad = ad_line.split(':')[1].strip() == 'TRUE'
    except StopIteration:
        includes_ad = False

    split = [line.strip().split() for line in data.strip().split('\n')]
    # transpose the list of lists for easier processing
    transposed = list(map(list, zip(*split)))

    # read in the first three columns as hkl
    collect_dict = {
        'refl_h': tuple(int(val) for val in transposed[0]),
        'refl_k': tuple(int(val) for val in transposed[1]),
        'refl_l': tuple(int(val) for val in transposed[2])
    }

    # split all data entries and assign them to the atom names
    data_entries = {
        scatterer: [float(val.split(',')[0]) + 1j * float(val.split(',')[1])
            for val in values]
        for scatterer, values in zip(scatterers, transposed[3:])
    } 

    # put everything in data entries into the collect_dict
    collect_dict.update(**data_entries)

    # Convert to pandas DataFrame for easy merging
    read_df = pd.DataFrame(collect_dict)

    # create the comparison DataFrame from the known hkl indexes
    resort_df = pd.DataFrame({
        'refl_h': index_vec_h[:, 0],
        'refl_k': index_vec_h[:, 1],
        'refl_l': index_vec_h[:, 2]
    })

    f0j = np.zeros((symm_mats_vecs[0].shape[0], len(labels), index_vec_h.shape[0]), dtype=np.complex128)

    # assign everything to the f0j_array
    vec_h_read = read_df[['refl_h', 'refl_k', 'refl_l']].values
    for symm_index, (symm_matrix, _) in enumerate(zip(*symm_mats_vecs)):
        #symm_df = read_df.copy()
        symm_df = resort_df.copy()
        hrot, krot, lrot = np.einsum('zx, xy -> zy', index_vec_h, symm_matrix).T 
        symm_df['refl_h'] = hrot
        symm_df['refl_k'] = krot
        symm_df['refl_l'] = lrot
        merge = pd.merge(symm_df, read_df, how='left', on=['refl_h', 'refl_k', 'refl_l'])
        values = merge[labels].values.copy()
        if not includes_ad:
            symm_df['refl_h'] = -hrot
            symm_df['refl_k'] = -krot
            symm_df['refl_l'] = -lrot
            merge = pd.merge(symm_df, read_df, how='left', on=['refl_h', 'refl_k', 'refl_l'])
            mask = np.logical_not(np.isfinite(values))
            values[mask] = np.conj(merge[labels].values[mask])
        f0j[symm_index] = values.T
    assert np.sum(np.logical_not(np.isfinite)) == 0, 'Some values were not found in the .tsc file, are all symmetry equivalent reflections present?'

    return f0j


def calc_f0j_core(
    cell_mat_m: np.ndarray,
    element_symbols: List[str],
    positions: np.ndarray,
    index_vec_h: np.ndarray,
    symm_mats_vecs: np.ndarray,
    computation_dict: Dict[str, Any]
) -> np.ndarray:
    raise NotImplementedError('This is currently not implemented')


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
    if 'cif_addition' in computation_dict:
        cif_addition = computation_dict['cif_addition']
    else:
        cif_addition = ''

    addition = f"""  - Refinement was done using structure factors
    derived from atomic form factors read from a .tsc file
{cif_addition}
"""
    return addition

