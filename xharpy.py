from jax.config import config
config.update('jax_enable_x64', True)

import jax.numpy as np
import pandas as pd
import os
from collections import OrderedDict, namedtuple
import warnings
import re
import jax
import numpy as onp
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.integrate import simps
import gpaw
import ase
from ase import Atoms
from ase.spacegroup import crystal
from ase.units import Bohr
#from gpaw.analyse.hirshfeld import HirshfeldPartitioning
from .hirshfeld_val import HirshfeldPartitioning
from .restraints import resolve_restraints


def ciflike_to_dict(filename, return_only_loops=False, resolve_std=True):
    """
    Takes the filename of a ciflike file such as the cif format itself or fco files
    and returns the content as an ordered dictionary.
    
    All tables, that are contained in the loop(ed) sections are returned under the 
    'loops' keyword as pandas dataframes. Everything else is stored under the keyword
    in cif minus the leading underscore. Will work with multiple structures in one cif.
    That is why the first dict has data names as keys and the actual other keywords as
    a contained dict.
    
    Has two keyword options:
    return_only_loops: [False] will only return the contained loops
    resolve_std: [True] will introduce new keywords with _std for all values containing
                 errors such as 11(2). Also works on the dataframes.
    
    """
    PATTERN = re.compile(r'''((?:[^ "']|"[^"]*"|'[^']*')+)''')
    with open(filename, 'r') as fo:
        lines = [line[:-1] for line in fo.readlines()]
    datablocks = OrderedDict()
    current_loop_lines = []
    current_loop_titles = []
    current_block = 'preblock'
    in_loop = False
    in_loop_titles = False
    current_line_collect = []
    for index, raw_line in enumerate(lines):
        line = raw_line.strip().lstrip()
        if line == '' or index == len(lines) - 1:
            if in_loop:
                in_loop = False
                new_df = pd.DataFrame(current_loop_lines)
                for key in new_df:
                    new_df[key] = pd.to_numeric(new_df[key], errors='ignore')
                if resolve_std:
                    for column in new_df.columns:
                        if new_df[column].dtype != 'O':
                            continue
                        concatenate = ''.join(new_df[column])
                        if  re.search(r'[\(\)]', concatenate) is not None and re.search(r'[^\d^\.^\(^\)\-\+]', concatenate) is None:
                            values, errors = np.array([split_error(val) for val in new_df[column]]).T
                            new_df[column] = values
                            new_df[column+'_std'] = errors
                datablocks[current_block]['loops'].append(new_df)
                current_loop_lines = []
                current_loop_titles = []
                current_line_collect = []
            continue
        if in_loop and not in_loop_titles and line.startswith('_') or line.startswith('loop_'):
            in_loop = False
            if len(current_loop_lines) > 0:
                new_df = pd.DataFrame(current_loop_lines)
                for key in new_df:
                    new_df[key] = pd.to_numeric(new_df[key], errors='ignore')
                if resolve_std:
                    for column in new_df.columns:
                        if new_df[column].dtype != 'O':
                            continue
                        concatenate = ''.join(new_df[column])
                        if  re.search(r'[\(\)]', concatenate) is not None and re.search(r'[^\d^\.^\(^\)\-\+]', concatenate) is None:
                            values, errors = np.array([split_error(val) for val in new_df[column]]).T
                            new_df[column] = values
                            new_df[column+'_std'] = errors
                datablocks[current_block]['loops'].append(new_df)
            current_loop_lines = []
            current_loop_titles = []
            current_line_collect = []
        if line[0] == '#':
            continue
        elif line[:5] == 'data_':
            current_block = line[5:]
            datablocks[current_block] = OrderedDict([('loops', [])])
        elif line[:5] == 'loop_':
            in_loop = True
            in_loop_titles = True
        elif in_loop and in_loop_titles and line[0] == '_':
            current_loop_titles.append(line[1:])
        elif in_loop:
            in_loop_titles = False
            line_split = [item.strip() for item in PATTERN.split(line) if item != '' and not item.isspace()]
            line_split = [item[1:-1] if "'" in item else item for item in line_split]
            current_line_collect += line_split
            if len(current_line_collect) == len(current_loop_titles):
                current_loop_lines.append(OrderedDict())
                for index2, item in enumerate(current_line_collect):
                    current_loop_lines[-1][current_loop_titles[index2]] = item
                current_line_collect = []
        elif line[0] == '_':
            line_split = [item.strip() for item in PATTERN.split(line) if item != '' and not item.isspace()]
            line_split = [item[1:-1] if "'" in item else item for item in line_split]
            if len(line_split) > 1:
                if resolve_std:
                    test = line_split[1]
                    if re.search(r'[^\d]', test) is None:
                        datablocks[current_block][line_split[0][1:]] = int(test)
                    elif re.search(r'[^\d^\.]', test) is None and re.search(r'\d', test) is not None:
                        datablocks[current_block][line_split[0][1:]] = float(test)
                    elif re.search(r'[\(\)]', test) is not None and re.search(r'[^\d^\.^\(^\)\-\+]', test) is None:
                        val, error = split_error(test)
                        datablocks[current_block][line_split[0][1:]] = val
                        datablocks[current_block][line_split[0][1:] + '_std'] = error
                    else:
                        datablocks[current_block][line_split[0][1:]] = line_split[1]
                else:
                    datablocks[current_block][line_split[0][1:]] = line_split[1]
    if return_only_loops:
        loops = []
        for value in datablocks.values():
            for loop in value['loops']:
                loops.append(loop)
        return loops
    else:
        return datablocks

def split_error(string):
    """
    Helper function to split a string containing a value with error in brackets
    to a single value.
    """
    int_search = re.search(r'([\-\d]*)\((\d*)\)', string)
    search = re.search(r'(\-{0,1})([\d]*)\.(\d*)\((\d*)\)', string)
    if search is not None:
        sign, before_dot, after_dot, err = search.groups()
        if sign == '-':
            return -1 * (int(before_dot) + int(after_dot) * 10**(-len(after_dot))), int(err) * 10**(-len(after_dot))
        else:
            return int(before_dot) + int(after_dot) * 10**(-len(after_dot)), int(err) * 10**(-len(after_dot))
    elif int_search is not None:
        value, error = int_search.groups()
        return int(value), int(error)
    else:
        return float(string), 0.0  
    
def cell_constants_to_M(a, b, c, alpha, beta, gamma):
    """
    Generates a matrix with the three lattice vectors as lines
    unit of length will be as given by the cell constants
    """
    alpha = alpha / 180.0 * np.pi
    beta = beta / 180.0 * np.pi
    gamma = gamma / 180.0 * np.pi
    M = np.array(
        [
            [
                a,
                0,
                0
            ],
            [
                b * np.cos(gamma),
                b * np.sin(gamma),
                0
            ],
            [
                c * np.cos(beta),
                c * (np.cos(alpha) - np.cos(gamma) * np.cos(beta)) / np.sin(gamma),
                c / np.sin(gamma) * np.sqrt(1.0 - np.cos(alpha)**2 - np.cos(beta)**2
                                            - np.cos(gamma)**2
                                            + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
            ]
        ]
    )
    return M.T


def symm_to_matrix_vector(instruction):
    """
    Converts a instruction such as -x, -y, 0.5+z to a symmetry matrix and a 
    translation vector
    """
    instruction_strings = [val.replace(' ', '').upper() for val in instruction.split(',')]
    matrix = np.zeros((3,3), dtype=np.float64)
    vector = np.zeros(3, dtype=np.float64)
    for xyz, element in enumerate(instruction_strings):
        # search for fraction in a/b notation
        fraction1 = re.search(r'(-{0,1}\d{1,3})/(\d{1,3})(?![XYZ])', element)
        # search for fraction in 0.0 notation
        fraction2 = re.search(r'(-{0,1}\d{0,1}\.\d{1,4})(?![XYZ])', element)
        # search for whole numbers
        fraction3 = re.search(r'(-{0,1}\d)(?![XYZ])', element)
        if fraction1:
            vector = jax.ops.index_update(vector, jax.ops.index[xyz], float(fraction1.group(1)) / float(fraction1.group(2)))
        elif fraction2:
            vector = jax.ops.index_update(vector, jax.ops.index[xyz], float(fraction2.group(1)))
        elif fraction3:
            vector = jax.ops.index_update(vector, jax.ops.index[xyz], float(fraction3.group(1)))

        symm = re.findall(r'-{0,1}[\d\.]{0,8}[XYZ]', element)
        for xyz_match in symm:
            if len(xyz_match) == 1:
                sign = 1
            elif xyz_match[0] == '-' and len(xyz_match) == 2:
                sign = -1
            else:
                sign = float(xyz_match[:-1])
            if xyz_match[-1] == 'X':
                matrix = jax.ops.index_update(matrix, jax.ops.index[xyz, 0], sign)
            if xyz_match[-1] == 'Y':
                matrix = jax.ops.index_update(matrix, jax.ops.index[xyz, 1], sign)
            if xyz_match[-1] == 'Z':
                matrix = jax.ops.index_update(matrix, jax.ops.index[xyz, 2], sign)
    return matrix, vector


def fj_gpaw(cell_mat_m, element_symbols, positions, index_vec_h, symm_mats_vecs, gpaw_dict=None, restart=None, save='gpaw.gpw', explicit_core=True):
    """
    Calculate the aspherical atomic form factors from a density grid in the python package gpaw
    for each reciprocal lattice vector present in index_vec_h.
    """
    if gpaw_dict is None:
        gpaw_dict = {'xc': 'PBE', 'txt': 'gpaw.txt', 'h': 0.15, 'setups': 'paw'}
    else:
        gpaw_dict = gpaw_dict.copy()
    if 'gridrefinement' in gpaw_dict:
        gridrefinement = gpaw_dict['gridrefinement']
        #print(f'gridrefinement set to {gridrefinement}')
        del(gpaw_dict['gridrefinement'])
    else:
        gridrefinement = 2
    if 'average_symmequiv' in gpaw_dict:
        average_symmequiv = gpaw_dict['average_symmequiv']
        #print(f'average symmetry equivalents: {average_symmequiv}')
        del(gpaw_dict['average_symmequiv'])
    else:
        average_symmequiv = False
    symm_positions, symm_symbols, inv_indexes = expand_symm_unique(element_symbols,
                                                                   positions,
                                                                   cell_mat_m,
                                                                   symm_mats_vecs)
    if restart is None:
        atoms = crystal(symbols=symm_symbols,
                        basis=symm_positions % 1,
                        cell=cell_mat_m.T)
        calc = gpaw.GPAW(**gpaw_dict)
        atoms.set_calculator(calc)
        e1 = atoms.get_potential_energy()
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                atoms, calc = gpaw.restart(restart, txt=gpaw_dict['txt'])
                atoms.set_scaled_positions(symm_positions % 1)
                e1 = atoms.get_potential_energy()
        except:
            print('  failed to load the density from previous calculation. Starting from scratch')
            atoms = crystal(symbols=symm_symbols,
                            basis=symm_positions % 1,
                            cell=cell_mat_m.T)
            calc = gpaw.GPAW(**gpaw_dict)
            atoms.set_calculator(calc)
            e1 = atoms.get_potential_energy()

    e1 = atoms.get_potential_energy()

    density = onp.array(calc.get_all_electron_density(gridrefinement=gridrefinement, skip_core=explicit_core))
    if explicit_core:
        n_elec = sum([setup.Z for setup in calc.setups]) - sum(setup.Nc for setup in calc.density.setups)
    else:
        n_elec = sum([setup.Z for setup in calc.setups])
    density *= n_elec / density.sum()
    if save is not None:
        calc.write(save, mode='all')

    print('  calculated density with energy', e1)

    partitioning = HirshfeldPartitioning(calc)
    partitioning.initialize()
    overall_hdensity = partitioning.hdensity.get_density(list(range(symm_positions.shape[0])), gridrefinement=gridrefinement, skip_core=explicit_core)[0]
    h, k, l = np.meshgrid(*map(lambda n: onp.fft.fftfreq(n, 1/n).astype(np.int64), density.shape), indexing='ij')
    f0j_indexes = inv_indexes.reshape((symm_mats_vecs[0].shape[0], positions.shape[0]))
    f0j = onp.zeros((symm_mats_vecs[0].shape[0], positions.shape[0], index_vec_h.shape[0]), dtype=np.complex128)

    if average_symmequiv:
        for atom_index, symm_atom_indexes in enumerate(f0j_indexes.T):
            f0j_sum = np.zeros_like(h, dtype=np.complex128)
            for symm_matrix, symm_atom_index in zip(symm_mats_vecs[0], symm_atom_indexes):
                h_density = density * partitioning.hdensity.get_density([symm_atom_index], gridrefinement=gridrefinement, skip_core=explicit_core)[0] / overall_hdensity
                frac_position = symm_positions[symm_atom_index]
                h_rot, k_rot, l_rot = onp.einsum('xy, y... -> x...', symm_matrix, np.array((h, k, l))).astype(np.int64)
                phase_to_zero = onp.exp(-2j * np.pi * (frac_position[0] * h + frac_position[1] * k + frac_position[2] * l))
                f0j_sum += (onp.fft.ifftn(h_density) * phase_to_zero * onp.prod(h.shape))[h_rot, k_rot, l_rot]
            f0j_sum /= len(symm_atom_indexes)

            for symm_index, symm_matrix in enumerate(symm_mats_vecs[0]):
                h_rot, k_rot, l_rot = onp.einsum('xy, zy -> zx', symm_matrix, index_vec_h).astype(np.int64).T
                f0j[symm_index, atom_index, :] = f0j_sum[h_rot, k_rot, l_rot]
    else:
        h_vec, k_vec, l_vec = index_vec_h.T
        already_known = {}
        for atom_index, symm_atom_indexes in enumerate(f0j_indexes.T):
            f0j_sum = np.zeros_like(h, dtype=np.complex128)
            for symm_index, (symm_matrix, symm_atom_index) in enumerate(zip(symm_mats_vecs[0], symm_atom_indexes)):
                if symm_atom_index in list(already_known.keys()):
                    equiv_symm_index, equiv_atom_index = already_known[symm_atom_index]
                    f0j[symm_index, atom_index, :] = f0j[equiv_symm_index, equiv_atom_index, :]
                else:
                    h_density = density * partitioning.hdensity.get_density([symm_atom_index], gridrefinement=gridrefinement, skip_core=explicit_core)[0] / overall_hdensity
                    frac_position = symm_positions[symm_atom_index]
                    phase_to_zero = onp.exp(-2j * np.pi * (frac_position[0] * h_vec + frac_position[1] * k_vec + frac_position[2] * l_vec))
                    f0j[symm_index, atom_index, :] = (onp.fft.ifftn(h_density) * onp.prod(h.shape))[h_vec, k_vec, l_vec] * phase_to_zero
                    already_known[symm_atom_index] = (symm_index, atom_index)

    return f0j


def f_core_from_spline(spline, g_k, k=13):
    r_max = spline.get_cutoff()
    r = onp.zeros(2**k + 1)
    r[1:] = onp.exp(-1 * np.linspace(1.25 * k, 0.0 , 2**k)) * r_max
    #r[-2**(k-1):]= np.exp(-1 * np.linspace(1.25 * k, 0.0, 2**(k-1))) * r_max
    #r = np.linspace(0.0, r_max * 1.01, 2**k + 1) 
    r[0] = 0
    gr = r[None,:] * g_k[:,None]
    j0 = onp.zeros_like(gr)
    j0[gr != 0] = onp.sin(2 * np.pi * gr[gr != 0]) / (2 * np.pi * gr[gr != 0])
    j0[gr == 0] = 1
    y00_factor = 0.5 * onp.pi**(-0.5)
    int_me = 4 * onp.pi * r**2  * spline.map(r) * j0
    return simps(int_me, x=r) * y00_factor


def calculate_f0j_core(cell_mat_m, element_symbols, positions, index_vec_h, symm_mats_vecs):
    symm_positions, symm_symbols, _ = expand_symm_unique(element_symbols,
                                                                   positions,
                                                                   cell_mat_m,
                                                                   symm_mats_vecs)
    atoms = crystal(symbols=symm_symbols,
                    basis=symm_positions % 1,
                    cell=cell_mat_m.T)
    calc = gpaw.GPAW(setups='paw', txt=None)
    atoms.set_calculator(calc)
    calc.initialize(atoms)
    cell_inv = onp.linalg.inv(atoms.cell.T)
    g_k3 = onp.einsum('xy, zy -> zx', cell_inv, index_vec_h)
    g_ks = onp.linalg.norm(g_k3, axis=-1)
    splines = {setup.symbol: setup.get_partial_waves()[:4] for setup in calc.density.setups}

    f0j_core = {}
    n_steps = 100
    n_per_step = 50

    for name, (_, _, nc, _) in list(splines.items()):
        if name in list(f0j_core.keys()):
            continue
        #if name == 'H':
        #    f0j_core[name] = onp.zeros_like(g_ks)
        if len(g_ks) > n_steps * n_per_step:
            print(f'  Calculating the core structure factor by spline for {name}')
            g_max = g_ks.max() * Bohr + 0.1
            #x_inv = np.linspace(-0.5, g_max, n_steps * n_per_step)
            k = np.log(n_steps * n_per_step)
            x_inv = onp.exp(-1 * np.linspace(1.25 * k, 0.0, n_steps * n_per_step)) * g_max 
            x_inv[0] = 0
            f0j = onp.zeros(n_steps * n_per_step)
            for index in range(n_steps):
               f0j[index * n_per_step:(index + 1) * n_per_step] = f_core_from_spline(nc, x_inv[index * n_per_step:(index + 1) * n_per_step], k=19) 
            f0j_core[name] = interp1d(x_inv, f0j, kind='cubic')(g_ks * Bohr)
        else:
            print(f'  Calculating the core structure factor for {name}')
            f0j = onp.zeros(len(g_ks))
            for index in range(n_per_step, len(g_ks) + n_per_step, n_per_step):
                start_index = index - n_per_step
                if index < len(g_ks):
                    end_index = index
                else:
                    end_index = len(g_ks)
                f0j[start_index:end_index] = f_core_from_spline(nc, g_ks[start_index:end_index] * Bohr, k=19)
            f0j_core[name] = f0j
    return np.array([f0j_core[symbol] for symbol in element_symbols])


def expand_symm_unique(type_symbols, coordinates, cell_mat_m, symm_mats_vec):
    #TODO: Make clean so that only position with itself is checked
    symm_mats_r, symm_vecs_t = symm_mats_vec
    pos_frac0 = coordinates % 1
    positions = onp.zeros((0, 3))
    type_symbols_symm = []
    for coords in (onp.einsum('axy, zy -> azx', symm_mats_r, pos_frac0)+ symm_vecs_t[:,None,:]) % 1:
        positions = onp.concatenate((positions, coords % 1))
        type_symbols_symm += type_symbols
    _, unique_indexes, inv_indexes = onp.unique(np.round(np.einsum('xy, zy -> zx', cell_mat_m, positions), 2), axis=0, return_index=True, return_inverse=True)
    return positions[unique_indexes,:].copy(), [type_symbols_symm[index] for index in unique_indexes], inv_indexes


@jax.jit
def calc_f(xyz, uij, cijk, dijkl, occupancies, index_vec_h, cell_mat_f, symm_mats_vecs, fjs):
    """Calculate the overall structure factors for given indexes of hkl"""
    
    #einsum indexes: k: n_symm, z: n_atom, h: n_hkl
    symm_mats_r, symm_vecs_t = symm_mats_vecs
    vec_S = -np.einsum('xy, zy -> zx', cell_mat_f.T, index_vec_h)
    vec_S_symm = np.einsum('kxy, zy -> kzx', symm_mats_r, vec_S)
    u_mats = uij[:, np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])]
    vib_factors = np.exp(-2 * np.pi**2 * np.einsum('kha, khb, zab -> kzh', vec_S_symm, vec_S_symm, u_mats))
    gram_charlier3_indexes = np.array([[[0, 3, 5], [3, 4, 9], [5, 9, 6]],
                                       [[3, 4, 9], [4, 1, 7], [9, 7, 8]],
                                       [[5, 9, 6], [9, 7, 8], [6, 8, 2]]])
    cijk_inner_sum = np.einsum('kha, khb, khc, zabc -> kzh',
                               vec_S_symm,
                               vec_S_symm,
                               vec_S_symm,
                               cijk[:, gram_charlier3_indexes])
    gram_charlier3 = (4.0j * np.pi**3 / 3) * cijk_inner_sum
    gram_charlier4_indexes = np.array([[[[0, 3, 5],    [3, 9, 12],   [5, 12, 10]],
                                        [[3, 9, 12],   [9, 4, 13],  [12, 13, 14]],
                                        [[5, 12, 10], [12, 13, 14], [10, 14, 6]]],
                                       [[[3, 9, 12],  [9, 4, 13],   [12, 13, 14]],
                                        [[9, 4, 13],   [4, 1, 7],   [13, 7, 11]],
                                        [[12, 13, 14], [13, 7, 11], [14, 11, 8]]],
                                       [[[5, 12, 10], [12, 13, 14], [10, 14, 6]],
                                        [[12, 13, 14], [13, 7, 11], [14, 11, 8]],
                                        [[10, 14, 6], [14, 11, 8], [6, 8, 2]]]])
    dijkl_inner_sum = np.einsum('kha, khb, khc, khd, zabcd -> kzh',
                                vec_S_symm,
                                vec_S_symm,
                                vec_S_symm,
                                vec_S_symm,
                                dijkl[:, gram_charlier4_indexes])
    gram_charlier4 = (2.0 * np.pi**2 / 3.0) * dijkl_inner_sum
    gc_factor = 1 - gram_charlier3 + gram_charlier4

    positions_symm = np.einsum('kxy, zy -> kzx', symm_mats_r, xyz) + symm_vecs_t[:, None, :]
    #uvw = positions_symm * n_gd[None, :]
    #uvw_0 = np.floor(uvw)
    #uvw_1 = uvw - uvw_0
    #phases0 = np.exp(-2j * np.pi * np.sum(uvw_0 / n_gd[None,:], axis=-1))[:, :, None]
    #phases_sum1 = ((1 - uvw_1[:, :, 0]) * (1 - uvw_1[:, :, 1]) * (1 - uvw_1[:, :, 2]))[:, :, None]
    #phases_sum2 = np.exp(-2j * np.pi * index_vec_h[:, 0] / n_gd[0])[None, None, :] * (uvw_1[:, :, 0] * (1 - uvw_1[:, :, 1]) * (1 - uvw_1[:, :, 2]))[:, :, None]
    #phases_sum3 = np.exp(-2j * np.pi * index_vec_h[:, 1] / n_gd[1])[None, None, :] * (uvw_1[:, :, 1] * (1 - uvw_1[:, :, 0]) * (1 - uvw_1[:, :, 2]))[:, :, None]
    #phases_sum4 = np.exp(-2j * np.pi * index_vec_h[:, 2] / n_gd[2])[None, None, :] * (uvw_1[:, :, 2] * (1 - uvw_1[:, :, 0]) * (1 - uvw_1[:, :, 1]))[:, :, None]
    #phases_sum5 = np.exp(-2j * np.pi * (index_vec_h[:, 0] / n_gd[0] + index_vec_h[:, 1] / n_gd[1]))[None, None, :] * (uvw_1[:, :, 0] * uvw_1[:, :, 1] * (1 - uvw_1[:, :, 2]))[:, :, None]
    #phases_sum6 = np.exp(-2j * np.pi * (index_vec_h[:, 0] / n_gd[0] + index_vec_h[:, 2] / n_gd[2]))[None, None, :] * (uvw_1[:, :, 0] * uvw_1[:, :, 2] * (1 - uvw_1[:, :, 1]))[:, :, None]
    #phases_sum7 = np.exp(-2j * np.pi * (index_vec_h[:, 1] / n_gd[1] + index_vec_h[:, 2] / n_gd[2]))[None, None, :] * (uvw_1[:, :, 1] * uvw_1[:, :, 2] * (1 - uvw_1[:, :, 0]))[:, :, None]
    #phases_sum8 = np.exp(-2j * np.pi * (index_vec_h[:, 0] / n_gd[0] + index_vec_h[:, 1] / n_gd[1]) + index_vec_h[:, 2] / n_gd[2])[None, None, :] * (uvw_1[:, :, 0] * uvw_1[:, :, 1] * uvw_1[:, :, 2])[:, :, None]
    #phases = phases0 * (phases_sum1 + phases_sum2 + phases_sum3 + phases_sum4 + phases_sum5 + phases_sum6 + phases_sum7 + phases_sum8)
    
    
    phases = np.exp(2j * np.pi * np.einsum('kzx, hx -> kzh', positions_symm, index_vec_h))
    structure_factors = np.sum(occupancies[None, :] *  np.einsum('kzh, kzh, kzh, kzh -> hz', phases,  vib_factors, fjs, gc_factor), axis=-1)
    #structure_factors = np.sum(occupancies[None, :] * vib_factors * np.einsum('hzk, kzh -> hz', phases, fjs), axis=-1)
    return structure_factors


def resolve_instruction(parameters, instruction):
    """Resolve fixed and refined parameters"""
    if type(instruction).__name__ == 'RefinedParameter':
        return_value = instruction.multiplicator * parameters[instruction.par_index] + instruction.added_value
    elif type(instruction).__name__ == 'FixedParameter':
        return_value = instruction.value
    return return_value


# construct the instructions for building the atomic parameters back from the linear parameter matrix
def create_construction_instructions(atom_table, constraint_dict, sp2_add, torsion_add, atoms_for_gc3, atoms_for_gc4, scaling0=1.0, refine_core=True):
    """
    Creates the instructions that are needed for reconstructing all atomic parameters from the refined parameters
    Additioally returns an initial guesss for the refined parameter list from the atom table.
    """
    parameters = np.full(10000, np.nan)
    current_index = 1
    parameters = jax.ops.index_update(parameters, jax.ops.index[0], scaling0)
    if refine_core:
        current_index += 1
        parameters = jax.ops.index_update(parameters, jax.ops.index[1], 1.0)
    construction_instructions = []
    known_torsion_indexes = {}
    names = atom_table['label']   
    for _, atom in atom_table.iterrows():
        if atom['label'] in sp2_add.keys():
            bound_atom, plane_atom1, plane_atom2, distance, occupancy = sp2_add[atom['label']]
            bound_index = np.where(names == bound_atom)[0][0]
            plane_atom1_index = np.where(names == plane_atom1)[0][0]
            plane_atom2_index = np.where(names == plane_atom2)[0][0]
            xyz_constraint = SingleTrigonalCalculated(bound_atom_index=bound_index,
                                                      plane_atom1_index=plane_atom1_index,
                                                      plane_atom2_index=plane_atom2_index,
                                                      distance=distance)
            adp_constraint = UEquivCalculated(atom_index=bound_index,
                                              multiplicator=1.2)
            construction_instructions.append(AtomInstructions(xyz=xyz_constraint,
                                                              uij=adp_constraint,
                                                              occupancy=FixedParameter(value=occupancy)))
            continue
        if atom['label'] in torsion_add.keys():
            bound_atom, angle_atom, torsion_atom, distance, angle, torsion_angle_add, group_index, occupancy = torsion_add[atom['label']]
            if group_index not in known_torsion_indexes.keys():
                known_torsion_indexes[group_index] = current_index
                parameters = jax.ops.index_update(parameters, jax.ops.index[current_index], torsion_add_starts[group_index])
                current_index += 1
            bound_index = np.where(names == bound_atom)[0][0]
            angle_index = np.where(names == angle_atom)[0][0]
            torsion_index = np.where(names == torsion_atom)[0][0]
            torsion_parameter = RefinedParameter(par_index=known_torsion_indexes[group_index],
                                                 multiplicator=1.0,
                                                 added_value=torsion_angle_add)
            xyz_constraint = TorsionCalculated(bound_atom_index=bound_index,
                                               angle_atom_index=angle_index,
                                               torsion_atom_index=torsion_index,
                                               distance=FixedParameter(value=distance),
                                               angle=FixedParameter(value=angle),
                                               torsion_angle=torsion_parameter)
            adp_constraint = UEquivCalculated(atom_index=bound_index,
                                              multiplicator=1.5)
            construction_instructions.append(AtomInstructions(xyz=xyz_constraint,
                                                              uij=adp_constraint,
                                                              occupancy=FixedParameter(value=occupancy)))
            continue

        xyz = atom[['fract_x', 'fract_y', 'fract_z']].values.astype(np.float64)
        if atom['label'] in constraint_dict.keys() and 'xyz' in constraint_dict[atom['label']].keys():
            constraint = constraint_dict[atom['label']]['xyz']
            instr_zip = zip(constraint.variable_indexes, constraint.multiplicators, constraint.added_value)
            xyz_instructions = tuple(RefinedParameter(par_index=current_index + par_index,
                                                      multiplicator=mult,
                                                      added_value=add) if par_index >= 0 
                                     else FixedParameter(value=add) for par_index, mult, add in instr_zip)
            n_pars = np.max(constraint.variable_indexes) + 1
            parameters = jax.ops.index_update(parameters, jax.ops.index[current_index:current_index + n_pars],
                                              [xyz[varindex] for index, varindex in zip(*np.unique(constraint.variable_indexes, return_index=True)) if index >=0])
            current_index += n_pars
        else:
            parameters = jax.ops.index_update(parameters, jax.ops.index[current_index:current_index + 3], list(xyz))
            xyz_instructions = tuple(RefinedParameter(par_index=array_index, multiplicator=1.0) for array_index in range(current_index, current_index + 3))
            current_index += 3

        adp = atom[['U_11', 'U_22', 'U_33', 'U_23', 'U_13', 'U_12']].values.astype(np.float64)
        if atom['label'] in constraint_dict.keys() and 'uij' in constraint_dict[atom['label']].keys():
            constraint = constraint_dict[atom['label']]['uij']
            if type(constraint).__name__ == 'ConstrainedValues':
                instr_zip = zip(constraint.variable_indexes, constraint.multiplicators, constraint.added_value) 
                adp_instructions = tuple(RefinedParameter(par_index=current_index + par_index,
                                                          multiplicator= mult,
                                                          added_value=add) if par_index >= 0 
                                         else FixedParameter(value=add) for par_index, mult, add in instr_zip)
                n_pars = np.max(constraint.variable_indexes) + 1

                parameters = jax.ops.index_update(parameters, jax.ops.index[current_index:current_index + n_pars],
                                                  [adp[varindex] for index, varindex in zip(*np.unique(constraint.variable_indexes, return_index=True)) if index >=0])
                current_index += n_pars
            elif type(constraint).__name__ == 'UEquivConstraint':
                bound_index = np.where(names == bound_atom)[0][0]
                adp_instructions = UEquivCalculated(atom_index=bound_index, multiplicator=constraint.multiplicator)
        else:
            parameters = jax.ops.index_update(parameters, jax.ops.index[current_index:current_index + 6], list(adp))
            adp_instructions = tuple(RefinedParameter(par_index=array_index, multiplicator=1.0) for array_index in range(current_index, current_index + 6))
            current_index += 6

        if 'C_111' in atom.keys():
            cijk = atom[['C111', 'C222', 'C333', 'C112', 'C122', 'C113', 'C133', 'C223', 'C233', 'C123']].values.astype(np.float64)
        else:
            cijk = np.zeros(10)

        if atom['label'] in constraint_dict.keys() and 'cijk' in constraint_dict[atom['label']].keys() and atom['label'] in atoms_for_gc3:
            constraint = constraint_dict[atom['label']]['cijk']
            instr_zip = zip(constraint.variable_indexes, constraint.multiplicators, constraint.added_value) 
            cijk_instructions = tuple(RefinedParameter(par_index=current_index + par_index,
                                                       multiplicator=mult,
                                                       added_value=add) if par_index >= 0 
                                      else FixedParameter(value=add) for par_index, mult, add in instr_zip)
            n_pars = np.max(constraint.variable_indexes) + 1

            parameters = jax.ops.index_update(parameters, jax.ops.index[current_index:current_index + n_pars],
                                              [cijk[varindex] for index, varindex in zip(*np.unique(constraint.variable_indexes, return_index=True)) if index >=0])
            current_index += n_pars
        elif atom['label'] in atoms_for_gc3:
            parameters = jax.ops.index_update(parameters, jax.ops.index[current_index:current_index + 10], list(cijk))
            cijk_instructions = tuple(RefinedParameter(par_index=array_index, multiplicator=1.0) for array_index in range(current_index, current_index + 10))
            current_index += 10
        else:
            cijk_instructions = tuple(FixedParameter(value=0.0) for index in range(10))

        if 'D_1111' in atom.keys():
            dijkl = atom[['D1111', 'D2222', 'D3333', 'D1112', 'D1222', 'D1113', 'D1333', 'D2223', 'D2333', 'D1122', 'D1133', 'D2233', 'D1123', 'D1223', 'D1233']].values.astype(np.float64)
        else:
            dijkl = np.zeros(15)

        if atom['label'] in constraint_dict.keys() and 'dijkl' in constraint_dict[atom['label']].keys() and atom['label'] in atoms_for_gc4:
            constraint = constraint_dict[atom['label']]['dijkl']
            instr_zip = zip(constraint.variable_indexes, constraint.multiplicators, constraint.added_value) 
            dijkl_instructions = tuple(RefinedParameter(par_index=current_index + par_index,
                                                        multiplicator=mult,
                                                        added_value=add) if par_index >= 0 
                                      else FixedParameter(value=add) for par_index, mult, add in instr_zip)
            n_pars = np.max(constraint.variable_indexes) + 1

            parameters = jax.ops.index_update(parameters, jax.ops.index[current_index:current_index + n_pars],
                                              [dijkl[varindex] for index, varindex in zip(*np.unique(constraint.variable_indexes, return_index=True)) if index >=0])
            current_index += n_pars
        elif atom['label'] in atoms_for_gc4:
            parameters = jax.ops.index_update(parameters, jax.ops.index[current_index:current_index + 15], list(dijkl))
            dijkl_instructions = tuple(RefinedParameter(par_index=array_index, multiplicator=1.0) for array_index in range(current_index, current_index + 15))
            current_index += 15
        else:
            dijkl_instructions = tuple(FixedParameter(value=0.0) for index in range(15))

        if atom['label'] in constraint_dict.keys() and 'occ' in constraint_dict[atom['label']].keys():
            constraint = constraint_dict[atom['label']]['occ']
            occupancy = FixedParameter(value=constraint.added_value[0])
        else:
            occupancy = FixedParameter(value=1.0)

        construction_instructions.append(AtomInstructions(
            name=atom['label'],
            element=atom['type_symbol'],
            dispersion_real = atom['type_scat_dispersion_real'],
            dispersion_imag = atom['type_scat_dispersion_imag'],
            xyz=xyz_instructions,
            uij=adp_instructions,
            cijk=cijk_instructions,
            dijkl=dijkl_instructions,
            occupancy=occupancy
        ))
    parameters = parameters[:current_index]
    return construction_instructions, parameters



def construct_values(parameters, construction_instructions, cell_mat_m):
    """Reconstruct xyz, adp-parameters and occupancies from the given construction instructions. Allows for the
    Flexible usage of combinations of fixed parameters and parameters that are refined, as well as constraints"""
    cell_mat_g = np.einsum('ja, jb -> ab', cell_mat_m, cell_mat_m)
    cell_mat_f = np.linalg.inv(cell_mat_m)
    cell_mat_g_star = np.einsum('ja, jb -> ab', cell_mat_f, cell_mat_f)
    #n_atoms = len(construction_instructions)
    xyz = np.array(
        [[resolve_instruction(parameters, inner_instruction) for inner_instruction in instruction.xyz]
          if type(instruction.xyz) in (tuple, list) else np.full(3, -9999.9) for instruction in construction_instructions]
    )
    uij = np.array(
        [[resolve_instruction(parameters, inner_instruction) for inner_instruction in instruction.uij]
          if type(instruction.uij) in (tuple, list) else np.full(6, -9999.9) for instruction in construction_instructions]
    )
    
    cijk = np.array(
        [[resolve_instruction(parameters, inner_instruction) for inner_instruction in instruction.cijk]
          if type(instruction.cijk) in (tuple, list) else np.full(6, -9999.9) for instruction in construction_instructions]
    )
    
    dijkl = np.array(
        [[resolve_instruction(parameters, inner_instruction) for inner_instruction in instruction.dijkl]
          if type(instruction.dijkl) in (tuple, list) else np.full(6, -9999.9) for instruction in construction_instructions]
    )
    occupancies = np.array([resolve_instruction(parameters, instruction.occupancy) for instruction in construction_instructions])    

    # second loop here for constructed options in order to have everything already available
    for index, instruction in enumerate(construction_instructions):
        # constrained positions
        if type(instruction.xyz).__name__ == 'TorsionCalculated':
            bound_xyz = cell_mat_m @ xyz[instruction.xyz.bound_atom_index]
            angle_xyz = cell_mat_m @ xyz[instruction.xyz.angle_atom_index]
            torsion_xyz = cell_mat_m @ xyz[instruction.xyz.torsion_atom_index]
            vec_ab = (angle_xyz - torsion_xyz)
            vec_bc_norm = -(bound_xyz - angle_xyz) / np.linalg.norm(bound_xyz - angle_xyz)
            distance = resolve_instruction(parameters, instruction.xyz.distance)
            angle = np.deg2rad(resolve_instruction(parameters, instruction.xyz.angle))
            torsion_angle = np.deg2rad(resolve_instruction(parameters, instruction.xyz.torsion_angle))
            vec_d2 = np.array([distance * np.cos(angle),
                               distance * np.sin(angle) * np.cos(torsion_angle),
                               distance * np.sin(angle) * np.sin(torsion_angle)])
            vec_n = np.cross(vec_ab, vec_bc_norm)
            vec_n = vec_n / np.linalg.norm(vec_n)
            rotation_mat_m = np.array([vec_bc_norm, np.cross(vec_n, vec_bc_norm), vec_n]).T
            xyz = jax.ops.index_update(xyz, jax.ops.index[index], cell_mat_f @ (rotation_mat_m @ vec_d2 + bound_xyz))

        if type(instruction.xyz).__name__ == 'SingleTrigonalCalculated':
            bound_xyz = xyz[instruction.xyz.bound_atom_index]
            plane1_xyz = xyz[instruction.xyz.plane_atom1_index]
            plane2_xyz = xyz[instruction.xyz.plane_atom2_index]
            addition = 2 * bound_xyz - plane1_xyz - plane2_xyz
            xyz = jax.ops.index_update(xyz, jax.ops.index[index], bound_xyz + addition / np.linalg.norm(cell_mat_m @ addition) * instruction.xyz.distance)
        
        # constrained displacements
        if type(instruction.uij).__name__ == 'UEquivCalculated':
            uij_parent = uij[instruction.uij.atom_index, [[0, 5, 4], [5, 1, 3], [4, 3, 2]]]
            uij = jax.ops.index_update(uij, jax.ops.index[index, :3], 1.0 / 3.0 * np.trace((cell_mat_g_star * uij_parent) @ cell_mat_g))
            uij = jax.ops.index_update(uij, jax.ops.index[index, 3:], 0.0)
    return xyz, uij, cijk, dijkl, occupancies

def resolve_instruction_esd(esds, instruction):
    """Resolve fixed and refined parameter esds"""
    if type(instruction).__name__ == 'RefinedParameter':
        return_value = instruction.multiplicator * esds[instruction.par_index]
    elif type(instruction).__name__ == 'FixedParameter':
        return_value = np.nan # One could pick zero, but this should indicate that an error is not defined
    return return_value

# TODO Write a constructs ESD function
def construct_esds(var_cov_mat, construction_instructions):
    # TODO Build analogous to the distance calculation function to get esds for all non-primitive calculations
    esds = np.sqrt(np.diag(var_cov_mat))
    xyz = np.array(
        [[resolve_instruction_esd(esds, inner_instruction) for inner_instruction in instruction.xyz]
          if type(instruction.xyz) in (tuple, list) else np.full(3, np.nan) for instruction in construction_instructions]
    )
    uij = np.array(
        [[resolve_instruction_esd(esds, inner_instruction) for inner_instruction in instruction.uij]
          if type(instruction.uij) in (tuple, list) else np.full(6, np.nan) for instruction in construction_instructions]
    )
    
    cijk = np.array(
        [[resolve_instruction_esd(esds, inner_instruction) for inner_instruction in instruction.cijk]
          if type(instruction.cijk) in (tuple, list) else np.full(6, np.nan) for instruction in construction_instructions]
    )
    
    dijkl = np.array(
        [[resolve_instruction_esd(esds, inner_instruction) for inner_instruction in instruction.dijkl]
          if type(instruction.dijkl) in (tuple, list) else np.full(6, np.nan) for instruction in construction_instructions]
    )
    occupancies = np.array([resolve_instruction_esd(esds, instruction.occupancy) for instruction in construction_instructions])
    return xyz, uij, cijk, dijkl, occupancies    

def calc_lsq_factory(cell_mat_m, symm_mats_vecs, index_vec_h, intensities_obs, stds_obs, construction_instructions, fjs_core=None):
    """Generates a calc_lsq function. Doing this with a factory function allows for both flexibility but also
    speed by automatic loop and conditional unrolling for all the stuff that is constant for a given structure."""
    construct_values_j = jax.jit(construct_values, static_argnums=(1,2))
    cell_mat_f = np.linalg.inv(cell_mat_m)
    def function(parameters, fjs):
        xyz, uij, cijk, dijkl, occupancies = construct_values_j(parameters, construction_instructions, cell_mat_m)
        if fjs_core is not None:
            fjs = parameters[1] * fjs + fjs_core[None, :, :]

        structure_factors = calc_f(
            xyz=xyz,
            uij=uij,
            cijk=cijk,
            dijkl=dijkl,
            occupancies=occupancies,
            index_vec_h=index_vec_h,
            cell_mat_f=cell_mat_f,
            symm_mats_vecs=symm_mats_vecs,
            fjs=fjs
        )
        intensities_calc = parameters[0] * np.abs(structure_factors)**2
        weights = 1 / stds_obs**2

        lsq = np.sum(weights * (intensities_obs - intensities_calc)**2) 
        return lsq
    return jax.jit(function)


def calc_var_cor_mat(cell_mat_m, symm_mats_vecs, index_vec_h, construction_instructions, intensities_obs, stds_obs, parameters, fjs, fjs_core=None):
    construct_values_j = jax.jit(construct_values, static_argnums=(1,2))
    cell_mat_f = np.linalg.inv(cell_mat_m)
    def function(parameters, fjs, index):
        xyz, uij, cijk, dijkl, occupancies = construct_values_j(parameters, construction_instructions, cell_mat_m)
        if fjs_core is not None:
            fjs = parameters[1] * fjs + fjs_core[None, :, :]

        structure_factors = calc_f(
            xyz=xyz,
            uij=uij,
            cijk=cijk,
            dijkl=dijkl,
            occupancies=occupancies,
            index_vec_h=index_vec_h[None, index],
            cell_mat_f=cell_mat_f,
            symm_mats_vecs=symm_mats_vecs,
            fjs=fjs[:, :, index, None]
        )
        return parameters[0] * np.abs(structure_factors[0])**2
    grad_func = jax.jit(jax.grad(function))

    collect = np.zeros((len(parameters), len(parameters)))
    for index, weight in enumerate(1 / stds_obs**2):
        val = grad_func(parameters, np.array(fjs), index)[:, None]
        collect += weight * (val @ val.T)

    lsq_func = calc_lsq_factory(cell_mat_m, symm_mats_vecs, index_vec_h, intensities_obs, stds_obs, construction_instructions, fjs_core)

    chi_sq = lsq_func(parameters, fjs) / (index_vec_h.shape[0] - len(parameters))

    return chi_sq * np.linalg.inv(collect)


### Internal Objects ####
AtomInstructions = namedtuple('AtomInstructions', [
    'name', # Name of atom, supposed to be unique
    'element', # Element symbol, e.g. 'Ca'
    'dispersion_real', # dispersion correction f'
    'dispersion_imag', # dispersion correction f''
    'xyz', # fractional coordinates
    'uij', # uij parameters U11, U22, U33, U23, U13, U12
    'cijk', # Gram Charlier 3rd Order C111, C222, C333, C112, C122, C113, C133, C223, C233, C123,
    'dijkl', # Gram Charlier 4th Order D1111, D2222, D3333, D1112, D1222, D1113, D1333, D2223, D2333, D1122, D1133, D2233, D1123, D1223, D1233
    'occupancy' # the occupancy
], defaults= [None, None, None, None, None, None, None, None])


RefinedParameter = namedtuple('RefinedParameter', [
    'par_index',     # index of the parameter in in the parameter array
    'multiplicator', # Multiplicator for the parameter. (default: 1.0)
    'added_value'    # Value that is added to the parameter (e.g. for occupancies)
], defaults=(None, 1, 0))


FixedParameter = namedtuple('FixedParameter', [
    'value'         # fixed value of the parameter 
])


UEquivCalculated = namedtuple('UEquivCalculated', [
    'atom_index',   # index of atom to set the U_equiv equal to 
    'multiplicator' # factor to multiply u_equiv with
])


SingleTrigonalCalculated = namedtuple('SingleTrigonalCalculated',[
    'bound_atom_index',  # index of atom the derived atom is bound to
    'plane_atom1_index', # first bonding partner of bound_atom
    'plane_atom2_index', # second bonding partner of bound_atom
    'distance'           # interatomic distance
])

TorsionCalculated = namedtuple('TorsionCalculated', [
    'bound_atom_index',   # index of  atom the derived atom is bound_to
    'angle_atom_index',   # index of atom spanning the given angle with bound atom
    'torsion_atom_index', # index of atom giving the torsion angle
    'distance',           # interatom distance
    'angle',              # interatom angle
    'torsion_angle'       # interatom torsion angle
])

### Objects for Use by the User

ConstrainedValues = namedtuple('ConstrainedValues', [
    'variable_indexes', # 0-x (positive): variable index; -1 means 0 -> not refined
    'multiplicators', # For higher symmetries mathematical conditions can include multiplicators
    'added_value', # Values that are added
])

UEquivConstraint = namedtuple('UEquivConstraint', [
    'bound_atom', # Name of the bound atom
    'multiplicator' # Multiplicator for UEquiv Constraint (Usually nonterminal: 1.2, terminal 1.5)
])


def har(cell_mat_m, symm_mats_vecs, hkl, construction_instructions, parameters, atom_fit_tol=1e-7, reload_step=2, gpaw_dict=None, explicit_core=True):
    """
    Basic Hirshfeld atom refinement routine. Will calculate the electron density on a grid spanning the unit cell
    First will refine the scaling factor. Afterwards all other parameters defined by the parameters, 
    construction_instructions pair will be refined until 10 cycles are done or the optimizer is converged fully
    """
    print('Preparing')
    index_vec_h = np.array(hkl[['h', 'k', 'l']].values.copy())
    type_symbols = [atom.element for atom in construction_instructions]
    constructed_xyz, *_ = construct_values(parameters,
                                                                                                                      construction_instructions,
                                                                                                                      cell_mat_m)

    dispersion_real = np.array([atom.dispersion_real for atom in construction_instructions])
    dispersion_imag = np.array([atom.dispersion_imag for atom in construction_instructions])
    f_dash = dispersion_real + 1j * dispersion_imag

    if explicit_core:
        f0j_core = calculate_f0j_core(cell_mat_m, type_symbols, constructed_xyz, index_vec_h, symm_mats_vecs)
        f0j_core += f_dash[:, None]
    else:
        f0j_core = None
    print('  building least squares function')
    calc_lsq = calc_lsq_factory(cell_mat_m, symm_mats_vecs, np.array(hkl[['h', 'k', 'l']]), np.array(hkl['intensity']), np.array(hkl['stderr']), construction_instructions, f0j_core)
    print('  setting up gradients')
    grad_calc_lsq = jax.jit(jax.grad(calc_lsq))
    #hess_calc_lsq = jax.jacfwd(grad_calc_lsq)

    print('step 0: calculating first atomic form factors')
    if reload_step == 0:
        restart = 'save.gpw'
    else:
        restart = None

    def minimize_scaling(x, fjs, parameters):
        for index, value in enumerate(x):
            parameters_new = jax.ops.index_update(parameters, jax.ops.index[index], value)
        return calc_lsq(parameters_new, fjs), grad_calc_lsq(parameters_new, fjs)[:len(x)]
    fjs = fj_gpaw(cell_mat_m,
                  type_symbols,
                  constructed_xyz,
                  index_vec_h,
                  symm_mats_vecs,
                  gpaw_dict=gpaw_dict,
                  save='save.gpw',
                  restart=restart,
                  explicit_core=explicit_core)
    if not explicit_core:
        fjs += f_dash[None,:,None]
    print('  Optimizing scaling')
    x = minimize(minimize_scaling, args=(fjs, parameters.copy()), x0=parameters[0], jac=True, options={'gtol': 1e-6 * index_vec_h.shape[0]})
    for index, val in enumerate(x.x):
        parameters = jax.ops.index_update(parameters, jax.ops.index[index], val)
    print(f'  wR2: {np.sqrt(x.fun / np.sum(hkl["intensity"].values**2 / hkl["stderr"].values**2)):8.6f}, nit: {x.nit}, {x.message}')

    r_opt_density = 1e10
    for refine in range(20):
        print(f'  calculating least squares sum')
        x = minimize(calc_lsq,
                     parameters,
                     jac=grad_calc_lsq,
                     #hess=hess_calc_lsq,
                     method='BFGS',
                     #method='trust-exact',
                     args=(fjs),
                     options={'gtol': 1e-7 * np.sum(hkl["intensity"].values**2 / hkl["stderr"].values**2)})
        print(f'  wR2: {np.sqrt(x.fun / np.sum(hkl["intensity"].values**2 / hkl["stderr"].values**2)):8.6f}, nit: {x.nit}, {x.message}')
        parameters = np.array(x.x) 
        if x.nit == 0:
            break
        elif x.fun < r_opt_density - atom_fit_tol or refine < 10:
            r_opt_density = x.fun
            #parameters_min1 = np.array(x.x)
        else:
            break 
            
        constructed_xyz, *_ = construct_values(parameters, construction_instructions, cell_mat_m)
        if refine >= reload_step - 1:
            restart = 'save.gpw'  
        else:
            restart = None                                                                                                               
        print(f'step {refine + 1}: calculating new structure factors')
        fjs = fj_gpaw(cell_mat_m,
                      type_symbols,
                      constructed_xyz,
                      index_vec_h,
                      symm_mats_vecs,
                      restart=restart,
                      gpaw_dict=gpaw_dict,
                      save='save.gpw',
                      explicit_core=explicit_core)
        if not explicit_core:
            fjs += f_dash[None,:,None]
    print('Calculation finished. calculating variance-covariance matrix.')
    var_cor_mat = calc_var_cor_mat(cell_mat_m, symm_mats_vecs, index_vec_h, construction_instructions, np.array(hkl['intensity']), np.array(hkl['stderr']), parameters, fjs, f0j_core)
    if explicit_core:
        fjs_return = fjs + f0j_core[None, :, :]
    else:
        fjs_return = fjs
    return parameters, fjs_return, fjs, var_cor_mat


def distance_with_esd(atom1_name, atom2_name, construction_instructions, parameters, var_cov_mat, cell_par, cell_std):
    names = [instr.name for instr in construction_instructions]
    index1 = names.index(atom1_name)
    index2 = names.index(atom2_name)

    def distance_func(parameters, cell_par):
        cell_mat_m = cell_constants_to_M(*cell_par)
        constructed_xyz, *_ = construct_values(parameters, construction_instructions, cell_mat_m)
        coord1 = constructed_xyz[index1]
        coord2 = constructed_xyz[index2]

        return np.linalg.norm(cell_mat_m @ (coord1 - coord2))
    
    distance = distance_func(parameters, cell_par)

    jac1, jac2 = jax.grad(distance_func, [0, 1])(parameters, cell_par)

    esd = np.sqrt(jac1[None, :] @ var_cov_mat @ jac1[None, :].T + jac2[None,:] @ np.diag(cell_std) @ jac2[None,:].T)
    return distance, esd[0, 0]
