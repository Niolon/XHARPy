"""This module contains the core functionality, including the refinement
and its analysis from the XHARPy library.
"""
import warnings
import datetime
from typing import Callable, List, Dict, Tuple, Optional, Union, Any
import pandas as pd
from .common_jax import jax, jnp
import numpy as np
from functools import partial
from copy import deepcopy
import pickle
from scipy.optimize import minimize

from .defaults import get_parameter_index, get_value_or_default
from .structure.common import AtomInstructions
from .structure.construct import construct_values, construct_esds
from .restraints import resolve_restraints
from .conversion import calc_sin_theta_ov_lambda, ucif2ucart, cell_constants_to_M

@jax.jit
def calc_f(
    xyz: jnp.ndarray,
    uij: jnp.ndarray,
    cijk: jnp.ndarray,
    dijkl: jnp.ndarray,
    occupancies: jnp.ndarray,
    index_vec_h: jnp.ndarray,
    cell_mat_f: jnp.ndarray,
    symm_mats_vecs: Tuple[jnp.ndarray, jnp.ndarray],
    f0j: jnp.ndarray
) -> jnp.ndarray :
    """Given a set of parameters, calculate the structure factors for all given
    reflections

    Parameters
    ----------
    xyz : jnp.ndarray
        size (N,3) array of fractional coordinates for the atoms in the asymmetric 
        unit
    uij : jnp.ndarray
        size (N, 6) array of anisotropic displacement parameters (isotropic
        parameters have to be transformed to anitropic parameters). Parameters
        need to be in convention as used e.g. Shelxl or the cif as U.
        Order: U11, U22, U33, U23, U13, U12
    cijk : jnp.ndarray
        size (N, 10) array of hird-order Gram-Charlier parameters as defined
        in Inter. Tables of Cryst. B (2010): Eq 1.2.12.7. Order: C111, C222, 
        C333, C112, C122, C113, C133, C223, C233, C123
    dijkl : jnp.ndarray
        size (N, 15) array of fourth-order Gram-Charlier parameters as defined
        in Inter. Tables of Cryst. B (2010): Eq 1.2.12.7. Order: D1111, D2222,
        D3333, D1112, D1222, D1113, D_1333, D2223, D2333, D1122, D1133, D2233,
        D1123, D1223, D1233
    occupancies : jnp.ndarray
        size (N) array of atomic occupancies. Atoms on special positions need to have
        an occupancy of 1/multiplicity
    index_vec_h : jnp.ndarray
        size (H, 3) array of Miller indicees of observed reflections
    cell_mat_f : jnp.ndarray
        size (3, 3) array with the reciprocal lattice vectors (1/Angstroem) as row
        vectors
    symm_mats_vecs : Tuple[jnp.ndarray, jnp.ndarray]
        size (K, 3, 3) array of symmetry matrices and (K, 3) array of translation
        vectors for all symmetry elements in the unit cell
    f0j : jnp.ndarray
        size (K, N, H) array of atomic form factors for all reflections and symmetry
        generated atoms within the unit cells. Atoms on special positions are 
        present multiple times and have the atomic form factor of the full atom.

    Returns
    -------
    structure_factors: jnp.ndarray
        size (H)-sized array with complex structure factors for each reflection
    """

    #einsum indexes: k: n_symm, z: n_atom, h: n_hkl
    lengths_star = jnp.linalg.norm(cell_mat_f, axis=0)
    symm_mats_r, symm_vecs_t = symm_mats_vecs

    vec_h_symm = jnp.einsum('zx, kxy -> kzy', index_vec_h, symm_mats_r) # vectorised version of H.T @ R
    u_mats = uij[:, jnp.array([[0, 5, 4],
                               [5, 1, 3],
                               [4, 3, 2]])]

    # Shelxl / Ucif convention
    vib_factors = jnp.exp(-2 * jnp.pi**2 * jnp.einsum('kha, khb, zab -> kzh',
                                                      vec_h_symm,
                                                      vec_h_symm,
                                                      u_mats * jnp.outer(lengths_star, lengths_star)))

    gram_charlier3_indexes = jnp.array([[[0, 3, 5], [3, 4, 9], [5, 9, 6]],
                                        [[3, 4, 9], [4, 1, 7], [9, 7, 8]],
                                        [[5, 9, 6], [9, 7, 8], [6, 8, 2]]])
    cijk_inner_sum = jnp.einsum('kha, khb, khc, zabc -> kzh',
                               vec_h_symm,
                               vec_h_symm,
                               vec_h_symm,
                               cijk[:, gram_charlier3_indexes])
    gram_charlier3 = (4.0j * jnp.pi**3 / 3) * cijk_inner_sum
    gram_charlier4_indexes = jnp.array([[[[0, 3, 5],    [3, 9, 12],   [5, 12, 10]],
                                         [[3, 9, 12],   [9, 4, 13],  [12, 13, 14]],
                                         [[5, 12, 10], [12, 13, 14], [10, 14, 6]]],
                                        [[[3, 9, 12],  [9, 4, 13],   [12, 13, 14]],
                                         [[9, 4, 13],   [4, 1, 7],   [13, 7, 11]],
                                         [[12, 13, 14], [13, 7, 11], [14, 11, 8]]],
                                        [[[5, 12, 10], [12, 13, 14], [10, 14, 6]],
                                         [[12, 13, 14], [13, 7, 11], [14, 11, 8]],
                                         [[10, 14, 6], [14, 11, 8], [6, 8, 2]]]])
    dijkl_inner_sum = jnp.einsum('kha, khb, khc, khd, zabcd -> kzh',
                                 vec_h_symm,
                                 vec_h_symm,
                                 vec_h_symm,
                                 vec_h_symm,
                                 dijkl[:, gram_charlier4_indexes])
    gram_charlier4 = (2.0 * jnp.pi**4 / 3.0) * dijkl_inner_sum
    gc_factor = 1 - gram_charlier3 + gram_charlier4

    positions_symm = jnp.einsum('kxy, zy -> kzx', symm_mats_r, xyz) + symm_vecs_t[:, None, :]
    phases = jnp.exp(2j * jnp.pi * jnp.einsum('kzx, hx -> kzh', positions_symm, index_vec_h))
    structure_factors = jnp.sum(occupancies[None, :] *  jnp.einsum('kzh, kzh, kzh, kzh -> hz', phases, vib_factors, f0j, gc_factor), axis=-1)
    return structure_factors



def calc_lsq_factory(
    cell_mat_m: jnp.ndarray,
    symm_mats_vecs: Tuple[jnp.ndarray, jnp.ndarray],
    index_vec_h: jnp.ndarray,
    intensities_obs: jnp.ndarray,
    weights: jnp.ndarray,
    construction_instructions: List[AtomInstructions],
    f0j_core: jnp.ndarray,
    refinement_dict: Dict[str, Any],
    wavelength: Optional[float],
    restraint_instr_ind: List[Any]=[]
) -> Callable:
    """Creates calc_lsq functions that can be used with jax.jit and jax.grad.
    This way we can account for the very different needs of structures, while
    keeping the code down the line relatively simple.

    Parameters
    ----------
    cell_mat_m : jnp.ndarray
        size (3, 3) array with the cell vectors as row vectors (Angstroem)
    symm_mats_vecs : Tuple[jnp.ndarray, jnp.ndarray]
        size (K, 3, 3) array of symmetry  matrices and (K, 3) array of translation
        vectors for all symmetry elements in the unit cell
    index_vec_h : jnp.ndarray
        size (H, 3) array of Miller indicees of observed reflections
    intensities_obs : jnp.ndarray
        size (H) array of observed reflection intensities
    weights : jnp.ndarray
        size (H) array of weights for the individual reflections
    construction_instructions : List[AtomInstructions]
        List of instructions for reconstructing the atomic parameters
    f0j_core : jnp.ndarray
        size (N, H) array of atomic core form factors calculated separately
    refinement_dict : Dict[str, Any]
        Dictionary that contains options for the refinement.
    wavelength : Optional[float]
        wavelength of radiation used for measurement in Angstroem
    restraint_instr_ind : List[Any], optional
        List of internal restraint options. Is still very early in development,
        probably buggy and should not be used for research at the moment, by 
        default []

    Returns
    -------
    calc_lsq: Callable
        function that return the sum of least squares for a given set of 
        parameters and atomic form factors

    Raises
    ------
    NotImplementedError
        Core treatment method not implemented
    """
    cell_mat_f = jnp.linalg.inv(cell_mat_m).T
    flack_parameter = get_parameter_index('flack', refinement_dict)
    core = get_value_or_default('core', refinement_dict)
    core_parameter = get_parameter_index('core', refinement_dict)
 
    extinction = get_value_or_default('extinction', refinement_dict)
    extinction_parameter = get_parameter_index('extinction', refinement_dict)
    if extinction == 'shelxl':
        assert wavelength is not None, 'Wavelength needs to be defined in refinement_dict for shelxl extinction'
        sintheta = jnp.linalg.norm(jnp.einsum('xy, zy -> zx', cell_mat_f, index_vec_h), axis=1) / 2 * wavelength
        sintwotheta = 2 * sintheta * jnp.sqrt(1 - sintheta**2)
        extinction_factors = 0.001 * wavelength**3 / sintwotheta

    tds = get_value_or_default('tds', refinement_dict)

    if tds == 'Zavodnik':
        tds_indexes = get_parameter_index('tds', refinement_dict)
        sin_th_ov_lam = jnp.linalg.norm(jnp.einsum('xy, zy -> zx', cell_mat_f, index_vec_h), axis=1) / 2

    #construct_values_j = jax.jit(construct_values, static_argnums=(1))
    construct_values_j = jax.jit(partial(
        construct_values,
        construction_instructions=construction_instructions,
        cell_mat_m=cell_mat_m
    ))

    def function(parameters, f0j):
        #xyz, uij, cijk, dijkl, occupancies = construct_values_j(parameters, construction_instructions, cell_mat_m)
        xyz, uij, cijk, dijkl, occupancies = construct_values_j(parameters)
        if core == 'scale':
            f0j = parameters[core_parameter] * f0j + f0j_core[None, :, :]
        elif core == 'constant':
            f0j = f0j + f0j_core[None, :, :]
        elif core == 'combine':
            pass
        else:
            raise NotImplementedError('core description not implemented in lsq_factory')

        structure_factors = calc_f(
            xyz=xyz,
            uij=uij,
            cijk=cijk,
            dijkl=dijkl,
            occupancies=occupancies,
            index_vec_h=index_vec_h,
            cell_mat_f=cell_mat_f,
            symm_mats_vecs=symm_mats_vecs,
            f0j=f0j
        )
        if extinction == 'none':
            intensities_calc = parameters[0] * jnp.abs(structure_factors)**2
            #restraint_addition = 0
        elif extinction == 'shelxl':
            i_calc0 = jnp.abs(structure_factors)**2
            intensities_calc = parameters[0] * i_calc0 / jnp.sqrt(1 + parameters[extinction_parameter] * extinction_factors * i_calc0)
        elif extinction == 'secondary':
            i_calc0 = jnp.abs(structure_factors)**2             
            intensities_calc = parameters[0] * i_calc0 / (1 + parameters[extinction_parameter] * i_calc0)
            #restraint_addition = 0

        if tds == 'Zavodnik':
            a_tds = parameters[tds_indexes[0]] 
            b_tds = parameters[tds_indexes[1]]
            intensities_calc = intensities_calc * (1 + a_tds * sin_th_ov_lam**2 + b_tds * sin_th_ov_lam**3)

  
        if flack_parameter is not None:
            structure_factors2 = calc_f(
                xyz=xyz,
                uij=uij,
                cijk=cijk,
                dijkl=dijkl,
                occupancies=occupancies,
                index_vec_h=-index_vec_h,
                cell_mat_f=cell_mat_f,
                symm_mats_vecs=symm_mats_vecs,
                f0j=f0j
            )
            if extinction == 'none':
                intensities_calc2 = parameters[0] * jnp.abs(structure_factors2)**2
                #restraint_addition = 0
            elif extinction == 'shelxl':
                i_calc02 = jnp.abs(structure_factors2)**2
                intensities_calc2 = parameters[0] * i_calc02 / jnp.sqrt(1 + parameters[extinction_parameter] * extinction_factors * i_calc02)
            elif extinction == 'secondary':
                i_calc02 = jnp.abs(structure_factors2)**2             
                intensities_calc2 = parameters[0] * i_calc02 / (1 + parameters[extinction_parameter] * i_calc02)
                #restraint_addition = 0

            if tds == 'Zavodnik':
                intensities_calc2 = intensities_calc2 * (1 + a_tds * sin_th_ov_lam**2 + b_tds * sin_th_ov_lam**3)

            lsq = jnp.sum(weights * (intensities_obs - parameters[flack_parameter] * intensities_calc2 - (1 - parameters[flack_parameter]) * intensities_calc)**2) 
        else:
            lsq = jnp.sum(weights * (intensities_obs - intensities_calc)**2) 
        return lsq * (1 + resolve_restraints(xyz, uij, restraint_instr_ind, cell_mat_m) / (len(intensities_obs) - len(parameters)))
    return jax.jit(function)

def calc_var_cor_mat(
    cell_mat_m: jnp.ndarray,
    symm_mats_vecs: Tuple[jnp.ndarray, jnp.ndarray],
    index_vec_h: jnp.ndarray,
    intensities_obs: jnp.ndarray ,
    weights: jnp.ndarray,
    construction_instructions: List[AtomInstructions],
    f0j_core: jnp.ndarray,
    parameters: jnp.ndarray,
    f0j: jnp.ndarray,
    refinement_dict: Dict[str, Any],
    wavelength: Optional[float] = None
) -> jnp.ndarray:
    """Calculates the variance-covariance matrix for a given set of parameters
    At the moment is pretty slow, as it is not parallelised over reflections

    Parameters
    ----------
    cell_mat_m : jnp.ndarray
        size (3, 3) array with the cell vectors as row vectors (Angstroem)
    symm_mats_vecs : Tuple[jnp.ndarray, jnp.ndarray]
        size (K, 3, 3) array of symmetry matrices and (K, 3) array of translation 
        vectors for all symmetry elements in the unit cell
    index_vec_h : jnp.ndarray
        size (H, 3) array of Miller indicees of observed reflections
    intensities_obs : jnp.ndarray
        size (H) array of observed reflection intensities
    weights : jnp.ndarray
        size (H) array of weights for the individual reflections
    construction_instructions : List[AtomInstructions]
        List of instructions for reconstructing the atomic parameters
    f0j_core : jnp.ndarray
        size (N, H) array of atomic core form factors calculated separately
    parameters : jnp.ndarray
        size (P) array with the refined parameters
    f0j : jnp.ndarray
        size (K, N, H) array of atomic form factors for all reflections and symmetry
        generated atoms within the unit cells. Atoms on special positions are
        present multiple times and have the atomic form factor of the full atom.
    refinement_dict : Dict[str, Any]
        Dictionary that contains options for the refinement.
    wavelength : Optional[float], optional
        wavelength of radiation used for measurement in Angstroem, by default
        None

    Returns
    -------
    var_cov_mat: jnp.ndarray
        size (P, P) array containing the variance-covariance matrix

    Raises
    ------
    NotImplementedError
        Extinction method not implemented
    NotImplementedError
        Core treatment not implemented
    """   
    cell_mat_f = jnp.linalg.inv(cell_mat_m).T
    flack_parameter = get_parameter_index('flack', refinement_dict)
    core = get_value_or_default('core', refinement_dict)
    core_parameter = get_parameter_index('core', refinement_dict)
 
    extinction = get_value_or_default('extinction', refinement_dict)
    extinction_parameter = get_parameter_index('extinction', refinement_dict)
    if extinction == 'shelxl':
        assert wavelength is not None, 'Wavelength needs to be defined in refinement_dict for shelxl extinction'
        sintheta = jnp.linalg.norm(jnp.einsum('xy, zy -> zx', cell_mat_f, index_vec_h), axis=1) / 2 * wavelength
        sintwotheta = 2 * sintheta * jnp.sqrt(1 - sintheta**2)
        extinction_factors = 0.001 * wavelength**3 / sintwotheta

    tds = get_value_or_default('tds', refinement_dict)

    if tds == 'Zavodnik':
        tds_indexes = get_parameter_index('tds', refinement_dict)
        sin_th_ov_lam = jnp.linalg.norm(jnp.einsum('xy, zy -> zx', cell_mat_f, index_vec_h), axis=1) / 2
    
    #construct_values_j = jax.jit(construct_values, static_argnums=(1))
    construct_values_j = jax.jit(partial(
        construct_values,
        construction_instructions=construction_instructions,
        cell_mat_m=cell_mat_m
    ))

    def function(parameters, f0j, index):
        #xyz, uij, cijk, dijkl, occupancies = construct_values_j(
        #    parameters,
        #    construction_instructions,
        #    cell_mat_m
        #)
        xyz, uij, cijk, dijkl, occupancies = construct_values_j(
            parameters
        )
        if core == 'scale':
            f0j = parameters[core_parameter] * f0j + f0j_core[None, :, :]
        elif core == 'constant':
            f0j = f0j + f0j_core[None, :, :]
        elif core == 'combine':
            pass
        else:
            raise NotImplementedError('core description not implemented in lsq_factory')

        structure_factors = calc_f(
            xyz=xyz,
            uij=uij,
            cijk=cijk,
            dijkl=dijkl,
            occupancies=occupancies,
            index_vec_h=index_vec_h[None, index],
            cell_mat_f=cell_mat_f,
            symm_mats_vecs=symm_mats_vecs,
            f0j=f0j[:, :, index, None]
        )
        if extinction == 'none':
            intensities_calc = parameters[0] * jnp.abs(structure_factors)**2
            #restraint_addition = 0
        elif extinction == 'shelxl':
            i_calc0 = jnp.abs(structure_factors)**2
            intensities_calc = parameters[0] * i_calc0 / jnp.sqrt(1 + parameters[extinction_parameter] * extinction_factors[index] * i_calc0)
        elif extinction == 'secondary':
            i_calc0 = jnp.abs(structure_factors)**2             
            intensities_calc = parameters[0] * i_calc0 / (1 + parameters[extinction_parameter] * i_calc0)
            #restraint_addition = 0

        if tds == 'Zavodnik':
            a_tds = parameters[tds_indexes[0]] 
            b_tds = parameters[tds_indexes[1]]
            intensities_calc = intensities_calc * (1 + a_tds * sin_th_ov_lam**2 + b_tds * sin_th_ov_lam**3)

            
        if flack_parameter is not None:
            structure_factors2 = calc_f(
                xyz=xyz,
                uij=uij,
                cijk=cijk,
                dijkl=dijkl,
                occupancies=occupancies,
                index_vec_h=-index_vec_h[None, index],
                cell_mat_f=cell_mat_f,
                symm_mats_vecs=symm_mats_vecs,
                f0j=f0j[:, :, index, None]
            )
            if extinction == 'none':
                intensities_calc2 = parameters[0] * jnp.abs(structure_factors2)**2
                #restraint_addition = 0
            elif extinction == 'shelxl':
                i_calc02 = jnp.abs(structure_factors2)**2
                intensities_calc2 = parameters[0] * i_calc02 / jnp.sqrt(1 + parameters[extinction_parameter] * extinction_factors[index] * i_calc02)
            elif extinction == 'secondary':
                i_calc02 = jnp.abs(structure_factors2)**2             
                intensities_calc2 = parameters[0] * i_calc02 / (1 + parameters[extinction_parameter] * i_calc02)
                #restraint_addition = 0

            if tds == 'Zavodnik':
                intensities_calc2 = intensities_calc2 * (1 + a_tds * sin_th_ov_lam**2 + b_tds * sin_th_ov_lam**3)

            return parameters[flack_parameter] * intensities_calc2[0] - (1 - parameters[flack_parameter]) * intensities_calc[0]
        else:
            return intensities_calc[0]
    grad_func = jax.jit(jax.grad(function))

    collect = jnp.zeros((len(parameters), len(parameters)))

    # TODO: Figure out a way to make this more efficient
    for index, weight in enumerate(weights):
        val = grad_func(parameters, jnp.array(f0j), index)[:, None]
        collect += weight * (val @ val.T)

    lsq_func = calc_lsq_factory(
        cell_mat_m,
        symm_mats_vecs,
        index_vec_h,
        intensities_obs,
        weights,
        construction_instructions,
        f0j_core,
        refinement_dict,
        wavelength
    )
    chi_sq = lsq_func(parameters, jnp.array(f0j)) / (index_vec_h.shape[0] - len(parameters))

    return chi_sq * jnp.linalg.inv(collect)

def refine(
    cell: jnp.ndarray,
    symm_mats_vecs: Tuple[jnp.ndarray, jnp.ndarray], 
    hkl: pd.DataFrame, 
    construction_instructions: List[AtomInstructions], 
    parameters: jnp.ndarray, 
    wavelength: Optional[float] = None,
    refinement_dict: dict = {},
    computation_dict: dict = {} 
)-> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """Refinement routine. The routine will refine for the given intensities
    against wR2(F^2).

    Parameters
    ----------
    cell : jnp.ndarray
        array with the lattice constants (Angstroem, Degree)
    symm_mats_vecs : Tuple[jnp.ndarray, jnp.ndarray]
        size (K, 3, 3) array of symmetry matrices and (K, 3) array of translation
        vectors for all symmetry elements in the unit cell
    hkl : pd.DataFrame
        pandas DataFrame containing the reflection data. Needs to have at least
        five columns: h, k, l, intensity, weight. Alternatively weight can be 
        substituted by esd_int. If no weight column is available the weights
        will be calculated as 1/esd_int**2. Additional columns will be ignored
    construction_instructions : List[AtomInstructions]
        List of instructions for reconstructing the atomic parameters from the
        list of refined parameters
    parameters : jnp.ndarray
        Starting values for the list of parameters
    wavelength : Optional[float], optional
        Measurement wavelength in Angstroem. Currently only used for Shelxl 
        style extinction correction. Can be omitted otherwise, by default None
    refinement_dict : dict, optional
        Dictionary with refinement options, by default {}

        Available options are:
          - f0j_source: 
            Source of the atomic form factors. The computation_dict 
            will be passed on to this method. See the individual files in
            f0j_sources for more information, by default 'gpaw'
            Tested options: 'gpaw', 'iam', 'gpaw_mpi'
            Some limitations: 'gpaw_spherical'
            Still untested: 'qe'
          - reload_step:   
            Starting with this step the computation will try to reuse the 
            density, if this is implemented in the source, by default 1
          - core:
            If this is implemented in a f0j_source, it will integrate the 
            frozen core density on a spherical grid and only use the valence
            density for the updated atomic form factos options are 
            'combine', which will not treat the core density separately,
            'constant' which will integrate and add the core density without
            scaling parameter and 'scale' which will refine a scaling 
            parameter for the core density which might for systematic
            deviations due to a coarse valence density grid (untested!)
            By default 'constant'
          - extinction:
            Use an extinction correction. Options: 'none' -> no extinction
            correction, 'shelxl' use the (empirical) formula used by SHELXL 
            to correct to correct for extinction, 'secondary' see Giacovazzo
            et al. 'Fundmentals of Crystallography' (1992) p.97, by default
            'none'
          - flack:
            Refinement of the flack parameter. Because xHARPy does not merge at
            the moment this should not be considered really implemented,
            by default False
          - max_dist_recalc:
            If the max difference in atomic positions is under this value in 
            Angstroems, no new structure factors will be calculated, by
            default 1e-6
          - max_iter:
            Maximum of refinement cycles if convergence not reached, by 
            default: 100
          - min_iter:
            Minimum refinement cycles. The refinement will stop if the
            wR2 increases if the current cycle is higher than min_iter,
            by default 10
          - core_io:
            Expects a tuple where the first entry can be 'save', 'load', 'none'
            which is the action that is taken with the core density. The 
            second argument in the tuple is the filename, to which the core
            density is saved to or loaded from
          - cutoff:
            Expects a tuple of three values where the first two are strings and
            the last one is a float value. First string is a cutoff mode. 
            Currently available are 'none' where all reflections are used,
            'sin(theta)/lambda' where the cutoff is set according to a user
            given resolution, 'fraction(f0jval)' where the resolution cutoff is
            set to include a certain fraction of the mean absolute *valence* 
            atomic form factors. 'I/esd(I)' can be used for excluding values
            based on the value over estimated standard deviation
            The second string can be either be 'above' or 'below' and 
            denominates in which direction values will be excluded.
            The final value is the actual cutoff value.
          - restraints:
            Not fully implemented. Do not use at the moment, by default []

    computation_dict : dict, optional
        Dict with options that are passed on to the f0j_source. See the 
        individual calc_f0j functions for a more detailed description, by 
        default {}

    Returns
    -------
    parameters: jnp.ndarray
        final refined parameters.
    var_cov_mat: jnp.ndarray,
        variance covariance matrix of the final refinement step
    information: Dict[str, Any]
        Dictionary with additional information, some of this is needed for 
        cif output

    Raises
    ------
    NotImplementedError
        f0j_source not implemented
    NotImplementedError
        Unknown core description
    NotImplementedError
        Second point where the core description could be missing
    """
    start = datetime.datetime.now()
    print('Started refinement at ', start)
    cell_mat_m = cell_constants_to_M(*cell)
    computation_dict = deepcopy(computation_dict)
    print('Preparing')
    index_vec_h = jnp.array(hkl[['h', 'k', 'l']].values.copy())
    type_symbols = [atom.element for atom in construction_instructions]
    parameters = jnp.array(parameters)
    constructed_xyz, constructed_uij, *_ = construct_values(parameters, construction_instructions, cell_mat_m)

    dispersion_real = jnp.array([atom.dispersion_real for atom in construction_instructions])
    dispersion_imag = jnp.array([atom.dispersion_imag for atom in construction_instructions])
    f_dash = dispersion_real + 1j * dispersion_imag
    f0j_source = get_value_or_default('f0j_source', refinement_dict)
    if f0j_source == 'gpaw':
        from .f0j_sources.gpaw_source import calc_f0j, calc_f0j_core
    elif f0j_source == 'iam':
        from .f0j_sources.iam_source import calc_f0j, calc_f0j_core
    elif f0j_source == 'gpaw_spherical':
        from .f0j_sources.gpaw_spherical_source import calc_f0j, calc_f0j_core
    elif f0j_source == 'qe':
        from .f0j_sources.qe_source import calc_f0j, calc_f0j_core
    elif f0j_source == 'gpaw_mpi':
        from .f0j_sources.gpaw_mpi_source import calc_f0j, calc_f0j_core
    elif f0j_source == 'tsc_file':
        from .f0j_sources.tsc_file_source import calc_f0j, calc_f0j_core
    elif f0j_source == 'nosphera2_orca':
        from .f0j_sources.nosphera2_orca_source import calc_f0j, calc_f0j_core
    elif f0j_source == 'custom_function':
        from .f0j_sources.custom_function_source import calc_f0j, calc_f0j_core
    else:
        raise NotImplementedError('Unknown type of f0j_source')

    reload_step = get_value_or_default('reload_step', refinement_dict)

    restraints = get_value_or_default('restraints', refinement_dict)
    if len(restraints) > 0:
        warnings.warn('Restraints are still highly experimental, The current implementation did not reproduce SHELXL results. So do not use them for research right now!')

    core_io, core_file = get_value_or_default('core_io', refinement_dict)
    core = get_value_or_default('core', refinement_dict)
    if f0j_source in ('iam') and core != 'combine':
        warnings.warn('core description is not possible with this f0j source')
    if core in ('scale', 'constant'):
        if core_io == 'load':
            with open(core_file, 'rb') as fo:
                f0j_core = pickle.load(fo)
            if index_vec_h.shape[0] != f0j_core.shape[1]:
                raise ValueError('The loaded core atomic form factors do not match the number of reflections')
            print('  loaded core atomic form factors from disk')
        elif f0j_source == 'qe':
            f0j_core, computation_dict = calc_f0j_core(
                cell_mat_m,
                construction_instructions,
                parameters,
                index_vec_h,
                computation_dict
            )
            f0j_core = jnp.array(f0j_core)
        else:
            f0j_core = jnp.array(calc_f0j_core(
                cell_mat_m,
                construction_instructions,
                parameters,
                index_vec_h,
                symm_mats_vecs,
                computation_dict
            ))
        if core_io == 'save':
            with open(core_file, 'wb') as fo:
                pickle.dump(f0j_core, fo)
            print('  saved core atomic form factors to disk')
        f0j_core += f_dash[:, None]

    elif core == 'combine':
        f0j_core = None
    else:
        raise NotImplementedError('Choose either scale, constant or combine for core description')

    max_distance_diff = get_value_or_default('max_dist_recalc', refinement_dict)

    max_iter = get_value_or_default('max_iter', refinement_dict)

    min_iter = get_value_or_default('min_iter', refinement_dict)


    if 'weight' not in hkl.columns:
        hkl['weight'] = 1 / hkl['esd_int']**2

    print('  calculating first atomic form factors')
    if reload_step == 0:
        restart = True
    else:
        restart = False

    f0j = calc_f0j(cell_mat_m,
                   construction_instructions,
                   parameters,
                   index_vec_h,
                   symm_mats_vecs,
                   computation_dict=computation_dict,
                   restart=restart,
                   explicit_core=f0j_core is not None)
    if f0j_core is None:
        f0j += f_dash[None,:,None]
    xyz_density = constructed_xyz

    cutoff_mode, cutoff_direction, cutoff_value = get_value_or_default('cutoff', refinement_dict)

    if cutoff_mode == 'none':
        hkl['included'] = True
    elif cutoff_mode in ('sin(theta)/lambda', 'fraction(f0jval)'):
        sinthovlam = np.array(calc_sin_theta_ov_lambda(jnp.linalg.inv(cell_mat_m).T, index_vec_h))
        if cutoff_mode == 'fraction(f0jval)':
            assert core != 'combine', 'This needs a separately calculated valence density'
            sort_args = np.argsort(sinthovlam)
            f0j_bar = np.mean(np.abs(f0j), axis=(0, 1))[sort_args]
            first_position = np.argwhere(np.cumsum(f0j_bar)/np.sum(f0j_bar) >= cutoff_value)[0][0]
            cutoff = float(sinthovlam[sort_args][first_position])
            print(f'  determined a sin(theta)/lambda cutoff of {cutoff:7.5f}')
        else:
            cutoff = cutoff_value
        hkl['included'] = True
        if cutoff_direction == 'above':
            hkl.loc[sinthovlam >= cutoff, 'included'] = False
        elif cutoff_direction == 'below':
            hkl.loc[sinthovlam < cutoff, 'included'] = False
        else:
            raise NotImplementedError("For the cutoff direction choose either 'above' or 'below'")
        print(f'  including {sum(hkl["included"])} of {len(hkl)} reflections')
    elif cutoff_mode == 'I/esd(I)':
        i_ov_esd = hkl['intensity'].values / hkl['esd_int'].values
        if cutoff_direction == 'above':
            hkl['included'] = i_ov_esd < cutoff_value
        elif cutoff_direction == 'below':
            hkl['included'] = i_ov_esd >= cutoff_value
        print(f'  including {sum(hkl["included"])} of {len(hkl)} reflections')
    else:
        raise NotImplementedError("cutoff_mode has to be 'none', 'sin(theta)/lambda', 'fraction_f0jval' or 'I/esd(I)")



    print('  building least squares function')
    calc_lsq = calc_lsq_factory(cell_mat_m,
                                symm_mats_vecs,
                                jnp.array(hkl[['h', 'k', 'l']].values),
                                jnp.array(hkl['intensity'].values),
                                jnp.array(hkl['weight'].values * hkl['included'].values.astype(np.int64)),
                                construction_instructions,
                                f0j_core,
                                refinement_dict,
                                wavelength,
                                restraints)
    print('  setting up gradients')
    grad_calc_lsq = jax.jit(jax.grad(calc_lsq))

    def minimize_scaling(x, parameters):
        parameters_new = None
        for index, value in enumerate(x):
            parameters_new = parameters.at[index].set(value)
        return calc_lsq(parameters_new, f0j), grad_calc_lsq(parameters_new, f0j)[:len(x)]
    print('step 0: Optimizing scaling')
    x = minimize(minimize_scaling,
                 args=(jnp.array(parameters)),
                 x0=parameters[0],
                 jac=True,
                 options={'gtol': 1e-8 * jnp.sum(hkl["intensity"].values**2 / hkl["esd_int"].values**2)})
    for index, val in enumerate(x.x):
        parameters = parameters.at[index].set(val)
    print(f'  wR2: {np.sqrt(x.fun / np.sum(hkl["intensity"].values**2 / hkl["esd_int"].values**2)):8.6f}, number of iterations: {x.nit}')

    r_opt_density = 1e10
    for refine in range(max_iter):
        print(f'  minimizing least squares sum')
        x = minimize(calc_lsq,
                     parameters,
                     jac=grad_calc_lsq,
                     method='BFGS',
                     args=(jnp.array(f0j)),
                     options={'gtol': 1e-13 * jnp.sum(hkl["intensity"].values**2 / hkl["esd_int"].values**2)})
        print(f'  wR2: {np.sqrt(x.fun / np.sum(hkl["intensity"].values**2 / hkl["esd_int"].values**2)):8.6f}, number of iterations: {x.nit}')
        shift = parameters - x.x
        parameters = jnp.array(x.x) 

        #if x.nit == 0:
        #    break
        if x.fun < r_opt_density or refine < min_iter:
            r_opt_density = x.fun
        else:
            break
        #with open('save_par_model.pkl', 'wb') as fo:
        #    pickle.dump({
        #        'construction_instructions': construction_instructions,
        #        'parameters': parameters
        #    }, fo) 
        
        constructed_xyz, *_ = construct_values(parameters, construction_instructions, cell_mat_m)
        restart = refine >= reload_step - 1
        if np.max(np.linalg.norm(np.einsum('xy, zy -> zx', cell_mat_m, constructed_xyz - xyz_density), axis=-1)) > max_distance_diff:
            print(f'step {refine + 1}: calculating new structure factors')
            del(f0j)

            f0j = calc_f0j(cell_mat_m,
                           construction_instructions,
                           parameters,
                           index_vec_h,
                           symm_mats_vecs,
                           computation_dict=computation_dict,
                           restart=restart,
                           explicit_core=f0j_core is not None)
            if f0j_core is None:
                f0j += f_dash[None,:,None]
            xyz_density = constructed_xyz
        else:
            print(f'step {refine + 1}: atom_positions are converged. No new structure factor calculation.')
    print('Calculation finished. calculating variance-covariance matrix.')
    var_cov_mat = calc_var_cor_mat(cell_mat_m,
                                   symm_mats_vecs,
                                   index_vec_h,
                                   jnp.array(hkl['intensity'].values),
                                   jnp.array(hkl['weight'].values * hkl['included'].values.astype(np.int64)),
                                   construction_instructions,
                                   f0j_core,
                                   parameters,
                                   f0j,
                                   refinement_dict,
                                   wavelength)
    if core == 'constant':
        f0j_all = f0j + f0j_core[None, :, :]
    elif core == 'scale':
        core_parameter = get_parameter_index('core', refinement_dict)
        f0j_all = parameters[core_parameter] * f0j + f0j_core[None, :, :]
    elif core == 'combine':
        f0j_all = f0j
    else:
        raise NotImplementedError('The used core is not implemented at the end of the har function (calculation of f0j')
    shift_ov_su = shift / np.sqrt(np.diag(var_cov_mat))
    end = datetime.datetime.now()
    print('Ended refinement at ', end)

    additional_information = {
        'f0j_anom': f0j_all,
        'shift_ov_su': shift_ov_su,
        'start': start,
        'end': end,
        'wR2 (approx)': np.sqrt(x.fun / np.sum(hkl["intensity"].values**2 / hkl["esd_int"].values**2))
    }
    return parameters, var_cov_mat, additional_information

