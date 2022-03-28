"""
This module contains functions for calculating quality indicators for a
finished refinement.   
"""

from .core import AtomInstructions, calc_f, construct_values, get_value_or_default, get_parameter_index
import pandas as pd
from typing import Tuple, List, Dict, Any, Union
from .conversion import calc_sin_theta_ov_lambda, cell_constants_to_M, calc_sin_theta_ov_lambda
import numpy as np
import warnings


def calculate_quality_indicators(
    cell: np.ndarray,
    symm_mats_vecs: Tuple[np.ndarray, np.ndarray],
    hkl: pd.DataFrame,
    construction_instructions: List[AtomInstructions],
    parameters: np.ndarray,
    wavelength: float,
    refinement_dict: Dict[str, Any],
    information: Dict[str, Any]
) -> Dict[str, float]:
    """[summary]

    Parameters
    ----------
    cell : np.ndarray
        array with the lattice constants (Angstroem, Degree)
    symm_mats_vecs : Tuple[np.ndarray, np.ndarray]
        (K, 3, 3) array of symmetry matrices and (K, 3) array of translation
        vectors for all symmetry elements in the unit cell
    hkl : pd.DataFrame
        pandas DataFrame containing the reflection data. Needs to have at least
        five columns: h, k, l, intensity, esd_int, Additional columns will be
        ignored
    construction_instructions : List[AtomInstructions]
        List of instructions for reconstructing the atomic parameters from the
        list of refined parameters
    parameters : np.ndarray
        final refined parameters
    wavelength: float
        Measurement wavelength in Angstroem
    refinement_dict : Dict[str, Any]
        Dictionary with refinement options
    information : Dict[str, Any]
        Dictionary with additional information, obtained from the refinement.
        the atomic form factors will be read from this dict.

    Returns
    -------
    quality_dict : Dict[str, float]
        Dictionary with different quality indicators.
    """
    cell_mat_m = np.array(cell_constants_to_M(*cell))
    hkl = hkl.copy()

    index_vec_h = hkl[['h', 'k', 'l']].values
    intensities = hkl['intensity'].values
    esd_int = hkl['esd_int'].values
    f0j = np.array(information['f0j_anom'])
    cell_mat_f = np.linalg.inv(cell_mat_m).T
    xyz, uij, cijk, dijkl, occupancies = construct_values(parameters, construction_instructions, cell_mat_m)

    structure_factors = np.array(calc_f(
        xyz=xyz,
        uij=uij,
        cijk=cijk,
        dijkl=dijkl,
        occupancies=occupancies,
        index_vec_h=index_vec_h,
        cell_mat_f=cell_mat_f,
        symm_mats_vecs=symm_mats_vecs,
        f0j=f0j
    ))

    extinction = get_value_or_default('extinction', refinement_dict)

    if extinction == 'none':
        hkl['intensity'] = np.array(intensities / parameters[0])
        hkl['esd_int'] = np.array(esd_int / parameters[0])
    elif extinction == 'secondary':
        extinction_parameter = get_parameter_index('extinction', refinement_dict)
        i_calc0 = np.abs(structure_factors)**2
        hkl['intensity'] = np.array(intensities / parameters[0] * (1 + parameters[extinction_parameter] * i_calc0))
        hkl['esd_int'] = np.array(esd_int / parameters[0] * (1 + parameters[extinction_parameter] * i_calc0))
    elif extinction == 'shelxl':
        extinction_parameter = get_parameter_index('extinction', refinement_dict)
        i_calc0 = np.abs(structure_factors)**2
        sintheta = np.linalg.norm(np.einsum('xy, zy -> zx', cell_mat_f, index_vec_h), axis=1) / 2 * wavelength
        sintwotheta = 2 * sintheta * np.sqrt(1 - sintheta**2)
        extinction_factors = 0.001 * wavelength**3 / sintwotheta
        hkl['intensity'] = np.array(intensities / parameters[0] * np.sqrt(1 + parameters[extinction_parameter] * extinction_factors * i_calc0))
        hkl['esd_int'] = np.array(esd_int / parameters[0] * np.sqrt(1 + parameters[extinction_parameter] * extinction_factors * i_calc0))
    else:
        raise NotImplementedError('Extinction correction method is not implemented in fcf routine')

    intensities = hkl['intensity'].values
    esd_int = hkl['esd_int'].values

    f_obs = np.sign(intensities) * np.sqrt(np.abs(intensities))
    f_obs_safe = np.array(f_obs)
    f_obs_safe[f_obs_safe == 0] = 1e-20
    i_over_2sigma = intensities / esd_int > 2

    r_f = np.sum(np.abs(f_obs - np.abs(structure_factors))) / np.sum(np.abs(f_obs))
    r_f_strong = np.sum(np.abs(f_obs[i_over_2sigma] - np.abs(structure_factors[i_over_2sigma]))) / np.sum(np.abs(f_obs[i_over_2sigma]))
    r_f2 = np.sum(np.abs(intensities - np.abs(structure_factors)**2)) / np.sum(intensities)
    wr2_strong  = np.sqrt(np.sum(1/esd_int[i_over_2sigma]**2 * (intensities[i_over_2sigma] - np.abs(structure_factors[i_over_2sigma])**2)**2) / np.sum(1/esd_int[i_over_2sigma]**2 * intensities[i_over_2sigma]**2))
    wr2= np.sqrt(np.sum(1/esd_int**2 * (intensities - np.abs(structure_factors)**2)**2) / np.sum(1/esd_int**2 * intensities**2))
    gof = np.sqrt(np.sum(1/esd_int**2 * (intensities - np.abs(structure_factors)**2)**2) / (len(intensities) - len(parameters)))

    return {
            'R(F)': r_f,
            'R(F)(I>2s)': r_f_strong,
            'R(F^2)': r_f2,
            'wR(F^2)': wr2,
            'wR(F^2)(I>2s)': wr2_strong,
            'GOF': gof
    }


def calculate_drk(
    df: pd.DataFrame,
    bins: Union[List, int, float] = 0.05,
    equal_sized_bins: bool = False,
    cell : np.ndarray = None
) -> Dict[str, Any]:
    """Can calculate a DRK plot for a given pandas DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        pandas DataFrame containing the reflection data. Can be either 
        generated by the write_fcf function or generated from a written
        fcf file by extracting the table with the io.ciflike_to_dict function
        Expected columns are intensity, esd_int, either a f_calc or a 
        intensity_calc function or the cif equivalents
        if no column sint/lambda is given the three columns h, k, and l are also
        needed
    bins : Union[List, int, float], optional
        Parameter used to define the bins. If it is a float, this is the
        step size in sin(theta)/lambda, if it is an integer, this is the number
        of bins the data is supposed to distributed into. By passing a list, 
        the function will use this as inner bin borders, 0 and inf do not need to
        be given separately, by default 0.05
    equal_sized_bins : bool, optional
        If true bins needs to be an integer. In that case the binning will
        create bins with an equal number of reflections instead of an equal
        range in sin(theta)/lambda, by default False
    cell : np.ndarray, optional
        size (6) array containing the cell parameters. Needs to be passed if
        the sint/lambda column is not already in df to calculate the resolution,
        by default None

    Returns
    -------
    result_dict : Dict[str, Any]
        Dictionary with the resulting drk_plots as keys in a dictionary

    Raises
    ------
    TypeError
        The function has encountered an invalid value for bins
    """
    df = df.copy()
    df = df.rename({
        'refln_index_h': 'h',
        'refln_index_k': 'k',
        'refln_index_l': 'l',
        'refln_F_squared_meas': 'intensity',
        'refln_F_squared_sigma': 'esd_int',
        'refln_F_squared_calc': 'intensity_calc',
        'refln_F_calc': 'f_calc',
        'refln_phase_calc': 'phase_calc',
        'refln_sint/lambda': 'sint/lambda'
    }, axis=1)
    if 'intensity_calc' not in df.columns:
        df['intensity_calc'] = np.abs(df['f_calc'])**2
    if 'sint/lambda' not in df.columns:
        assert cell is not None, 'You either need to give cell or sint/lambda in the df'
        cell_mat_m = cell_constants_to_M(*cell)
        cell_mat_f = np.linalg.inv(cell_mat_m)
        df['sint/lambda'] = np.array(calc_sin_theta_ov_lambda(cell_mat_f, df[['h', 'k', 'l']].values))
    if equal_sized_bins:
        try:
            n_bins = int(bins)
        except (TypeError, ValueError):
            assert not isinstance(bins, str)
            raise TypeError('bins has to be an integer number ')
        splits = np.array_split(df.sort_values('sint/lambda'), bins)
        mean_res = [np.mean(split['sint/lambda']) for split in splits]
        drk = [np.sum(split['intensity']) / np.sum(split['intensity_calc']) for split in splits]
        count = [len(split) for split in splits]
    else:
        # non equal bins
        try:
            n_bins = int(bins)
            assert float(bins) == int(bins) # check if float
            # number of bins given as integer
            res_min = df['sint/lambda'].min()
            res_max = df['sint/lambda'].max()
            limits = np.linspace(res_min, res_max, n_bins + 1)
        except (TypeError, ValueError):
            try:
                assert not isinstance(bins, str) # this would
            except:
                raise TypeError('bins has to be either an integer number, a float number or a list-like')
        except AssertionError:
            # bins is float so a step size
            step = float(bins)
            lower = np.floor(df['sint/lambda'].min() / step) * step
            limits = np.arange(lower, df['sint/lambda'].max() + step, step)
            
        conditions = [np.logical_and(df['sint/lambda'] > lim_low, df['sint/lambda'] < lim_high) for lim_low, lim_high in zip(limits[:-1], limits[1:])]
        drk = [np.sum(df[condition]['intensity']) / np.sum(df[condition]['intensity_calc']) for condition in conditions]
        count = [len(df[condition]) for condition in conditions]
    result = {
        'Sum(F^2_Obs)/Sum(F^2_calc)': np.array(drk),
        'count': np.array(count)
    }
    if equal_sized_bins:
        result['mean resolution'] = np.array(mean_res)
    else:
        result['lower limit'] = np.array(limits[:-1])
        result['upper limit'] = np.array(limits[1:])
    return result


def generate_hm(
    fcf_path: str,
    map_factor: float = 1/3,
    level_step : float= 0.01
) -> Dict[str, np.ndarray]:
    """Generates a Henn-Meindl or normal probability plot from the given fcf6
    file. Relies on cctbx for the calculation of the difference electron
    density.

    Parameters
    ----------
    fcf_path : str
        Path to the fcf6 file.
    map_factor : float, optional
        map_factor passed to the fft_map function of cctbx, by default 1/3
    level_step : float, optional
        step size in e/Ang^3 for the determination of the Henn-Meindl Plot,
        by default 0.01

    Returns
    -------
    results : Dict[str, np.ndarray]
        levels and hm values as dictionary.

    Raises
    ------
    ModuleNotFoundError
        No cctbx installed
    ValueError
        fcf file is probably not fcf6 as it is missing phases
    """
    try: 
        from iotbx import reflection_file_reader
    except:
        raise ModuleNotFoundError('cctbx is needed for this feature')
    reader = reflection_file_reader.cif_reader(fcf_path)
    arrays = reader.build_miller_arrays()[next(iter(reader.build_miller_arrays()))]
    fobs = arrays['_refln_F_squared_meas'].f_sq_as_f()
    try:
        fcalc = arrays['_refln_F_calc']
    except ValueError:
        raise ValueError('No _refln_F_calc in fcf, an fcf6 is needed')
    diff = fobs.f_obs_minus_f_calc(1.0, arrays['_refln_F_calc'])
    diff_map = diff.fft_map(map_factor)
    diff_map.apply_volume_scaling()
    real = diff_map.real_map_unpadded()
    arr = real.as_numpy_array()
    levels = np.arange(-1.0, 1.0 + level_step, level_step)
    start_at = np.argwhere(levels < arr.min())[-1, 0]
    end_at = np.argwhere(levels > arr.max())[0, 0]
    sum_levels = np.zeros_like(levels)
    for level_index, level in enumerate(levels[start_at:end_at]):
        sum_levels[level_index + start_at] += np.sum(np.logical_and(arr[1:,:,:] < level, level < arr[:-1,:,:]))
        sum_levels[level_index + start_at] += np.sum(np.logical_and(arr[1:,:,:] > level, level > arr[:-1,:,:]))
        sum_levels[level_index + start_at] += np.sum(np.logical_and(arr[:,1:,:] < level, level < arr[:,:-1,:]))
        sum_levels[level_index + start_at] += np.sum(np.logical_and(arr[:,1:,:] > level, level > arr[:,:-1,:]))
        sum_levels[level_index + start_at] += np.sum(np.logical_and(arr[:,:,1:] < level, level < arr[:,:,:-1]))
        sum_levels[level_index + start_at] += np.sum(np.logical_and(arr[:,:,1:] > level, level > arr[:,:,:-1]))
    n_pairs = 0
    for index in range(3):
        pair_shape = np.array(arr.shape)
        pair_shape[index] -= 1
        n_pairs += 2 * np.prod(pair_shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = np.log10(sum_levels) / np.log10(n_pairs**(1/3))
    return {'levels': levels, 'df': df}

def calculate_egross(    
    fcf_path: str,
    map_factor: float = 1/3
) -> float:
    """Calculates the e_gross value. Relies on cctbx for the calculation of
    the difference electron density.

    Parameters
    ----------
    fcf_path : str
        Path to the fcf6 file.
    map_factor : float, optional
        map_factor passed to the fft_map function of cctbx, by default 1/3

    Returns
    -------
    e_gross : float
        The resulting e_gross value
    """
    try: 
        from iotbx import reflection_file_reader
    except:
        raise ModuleNotFoundError('cctbx is needed for this feature')
    reader = reflection_file_reader.cif_reader(fcf_path)
    arrays = reader.build_miller_arrays()[next(iter(reader.build_miller_arrays()))]
    fobs = arrays['_refln_F_squared_meas'].f_sq_as_f()
    try:
        fcalc = arrays['_refln_F_calc']
    except ValueError:
        raise ValueError('No _refln_F_calc in fcf, an fcf6 is needed')
    diff = fobs.f_obs_minus_f_calc(1.0, arrays['_refln_F_calc'])
    diff_map = diff.fft_map(map_factor)
    #diff_map.apply_volume_scaling()
    real = diff_map.real_map_unpadded()
    arr = real.as_numpy_array()
    return np.mean(np.abs(arr)) / 2

def diff_density_values(
    fcf_path: str,
    map_factor: float = 1/3
) -> Dict[str, float]:
    """Calculates the rho_max, rho_min and rhos_sigma for a given fcf6 file

    Parameters
    ----------
    fcf_path : str
        Path to the fcf6 file.
    map_factor : float, optional
        map_factor passed to the fft_map function of cctbx, by default 1/3

    Returns
    -------
    Dict[str, float]
        The three value as a dictionary
    """
    from iotbx import reflection_file_reader
    reader = reflection_file_reader.cif_reader(fcf_path)
    arrays = reader.build_miller_arrays()[next(iter(reader.build_miller_arrays()))]
    fobs = arrays['_refln_F_squared_meas'].f_sq_as_f()
    fcalc = arrays['_refln_F_calc']
    diff = fobs.f_obs_minus_f_calc(1.0, arrays['_refln_F_calc'])
    diff_map = diff.fft_map(map_factor)
    diff_map.apply_volume_scaling()
    stats = diff_map.statistics()
    return {'rho_max': stats.max(), 'rho_min': stats.min(), 'rho_sigma': stats.sigma()}

