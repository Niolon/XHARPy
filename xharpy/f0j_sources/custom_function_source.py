"""
This f0j_source is meant for development of new f0j_sources. You can pass the
three functions calc_f0j, calc_f0j_core and generate_cif_output, which
will be called in every refinement cycle. If you have successfully 
implemented a new f0j_source, please share the joy and add it to the 
library for everyone to use, at the latest after successful publication.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from ..core import AtomInstructions


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
    """Calculate the atomic form factor or atomic valence form factors using 
    a custom function given within the computation_dict['calc_f0j'] parameter

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
        contains the function(s) that are called to calculate the atomic form 
        factors in the custom refinement, as well as a computation dict for 
        any settings you want to pass to your function. This is meant for
        development. If you have build something working and nice, please 
        consider transferring your sources to an \*_source.py file and uploading 
        it to the XHARPy Repository, at least after you have published your 
        results.
        
          - calc_f0j (Callable): function that implements the arguments: 
            cell_mat_m, construction_instructions, parameters, index_vec_h, 
            symm_mats_vecs, computation_dict, restart and explicit_core and 
            returns the calculated f0j values as an array. Examples and the 
            explanation of the input parameters can be found in any \*_source.py
            file in the calc_f0j function and its docstring
          - calc_f0j_core (Callable, Optional): function that implements the 
            arguments: cell_mat_m, construction_instructions, parameters, 
            index_vec_h, symm_mats_vecs, computation_dict and returns the 
            f0j_core values for a spherical frozen core density. Will only be
            called once at the beginning of the refinement. Is optional if 
            the refinement_dict['core'] is set to 'combine' otherwise it will 
            throw a NotImplementError. Examples and the explanation of the input 
            parameters can be found in any \*_source.py file in the 
            calc_f0j_core function and its docstring.
          - generate_cif_output (Callable): functional with the argument 
            computation dict, which will be called during .cif generation to write
            an output of the methodology used to the .cif file.
          - inner_computation_dict (Dict): dictionary with options to pass on to 
            your functions. The equivalent of the computation_dict that is used
            for the f0j_sources in XHARPy
    
    restart : bool, optional
        If true, the calculation will be restarted from a previous calculation 
        if possible
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

    custom_function = computation_dict['calc_f0j']

    computation_dict_func = computation_dict['inner_computation_dict']

    return custom_function(
        cell_mat_m,
        construction_instructions,
        parameters,
        index_vec_h,
        symm_mats_vecs,
        computation_dict_func,
        restart,
        explicit_core
    )

def calc_f0j_core(
    cell_mat_m: np.ndarray,
    construction_instructions: List[AtomInstructions],
    parameters: np.ndarray,
    index_vec_h: np.ndarray,
    symm_mats_vecs: np.ndarray,
    computation_dict: Dict[str, Any]
) -> np.ndarray:
    """Calculate the core atomic form factors with the function and option 
    provided. Will raise a NotImplementedError if called without an implemented
    function

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
        contains options for the calculation. Should include the 'calc_f0j_core'
        keyword if this function is meant to be used.

    Returns
    -------
    f0j_core: np.ndarray
        size (N, H) array of atomic core form factors calculated separately
    """
    if 'calc_f0j_core' not in computation_dict:
        raise NotImplementedError('Explicit core calculation is activated, but no calc_f0j_core function is given. Set refinement_options["core"] to "combine" or implement a function')
    custom_function = computation_dict['calc_f0j_core']

    computation_dict_func = computation_dict['inner_computation_dict']

    return custom_function(
        cell_mat_m,
        construction_instructions,
        parameters,
        index_vec_h,
        symm_mats_vecs,
        computation_dict_func
    )
    
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
    custom_function = computation_dict['generate_cif_output']

    computation_dict_func = computation_dict['inner_computation_dict']

    return custom_function(computation_dict_func)