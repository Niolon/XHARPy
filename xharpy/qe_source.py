import numpy as np
from copy import deepcopy
import subprocess
import os
import time
import re
from . import cubetools

from scipy.interpolate import interp1d
from scipy.integrate import simps
import warnings
from .core import expand_symm_unique

mass_dict = {
    'H': 1.008,  'He': 4.0026, 'Li': 6.9675, 'Be': 9.0122, 'B': 10.814, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'F': 18.998, 'Ne': 20.18, 'Na': 22.99, 'Mg': 24.306, 'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.068,
    'Cl': 35.452, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996,
    'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38, 'Ga': 69.723, 'Ge': 72.63,
    'As': 74.922, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224,
    'Nb': 92.906, 'Mo': 95.95, 'Tc': np.nan, 'Ru': 101.07, 'Rh': 102.91, 'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41,
    'In': 114.82, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.6, 'I': 126.9, 'Xe': 131.29, 'Cs': 132.91, 'Ba': 137.33, 
    'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24, 'Pm': np.nan, 'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25,
    'Tb': 158.93, 'Dy': 162.5, 'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05, 'Lu': 174.97, 'Hf': 178.49, 
    'Ta': 180.95, 'W': 183.84, 'Re': 186.21, 'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08, 'Au': 196.97, 'Hg': 200.59, 
    'Tl': 204.39, 'Pb': 207.2, 'Bi': 208.98, 'Po': 209.98, 'At': np.nan, 'Rn': np.nan, 'Fr': np.nan, 'Ra': np.nan,
    'Ac': np.nan, 'Th': 232.04, 'Pa': 231.04, 'U': 238.03
}

def qe_entry_string(name, value, string_sign=True):
    if type(value) is str:
        if string_sign:
            entry_str = f"'{value}'"
        else:
            entry_str = value
    elif type(value) is float:
        entry_str = f'{value}'
    elif type(value) is int:
        entry_str = f'{value}'
    elif type(value) is bool:
        if value:
            entry_str = '.TRUE.'
        else:
            entry_str = '.FALSE.'
    else:
        print(value, type(value))
        raise NotImplementedError(f'{type(value)} is not implemented')
    return f'    {name} = {entry_str}'

def qe_pw_file(symm_symbols, symm_positions, cell_mat_m, options_dict):
    qe_options = {
        'control': {
            'prefix': 'scf',
            'calculation': 'scf',
            'restart_mode': 'from_scratch',
            'pseudo_dir': '/home/niklas/qe_pseudo/',
            'outdir': './scratch',
        },
        'system' : {
            'ibrav': 0
        },
        'electrons': {
            'conv_thr': 1e-8,
            'mixing_beta': 0.2
        }
    }
    for section, secval in options_dict.items():
        if section in ('paw_files', 'core_electrons', 'k_points'):
            # these are either given in tables in QE or not used
            continue
        if section in qe_options:
            qe_options[section].update(secval)
        else:
            qe_options[section] = secval
    output = []
    unique_symbols = set(symm_symbols)
    assert 'paw_files' in options_dict, "'paw_files' entry in options_dict is missing"
    lines = []
    for symbol in unique_symbols:
        assert symbol in options_dict['paw_files'], f"No paw file (e.g. .UPF) given for {symbol} in options_dict['paw_files']"
        pp = options_dict['paw_files'][symbol]
        mass = mass_dict[symbol]
        lines.append(f'{symbol} {mass:8.3f}  {pp}')
    pp_string = '\n'.join(lines)
    output.append('ATOMIC_SPECIES\n' + pp_string)
    cell_mat_str = '\n'.join(f'{line[0]:9.6f} {line[1]:9.6f} {line[2]:9.6f}' for line in cell_mat_m.T)
    output.append('CELL_PARAMETERS angstrom\n' + cell_mat_str)
    atoms_string = '\n'.join(f'{sym} {pos[0]:12.10f} {pos[1]:12.10f} {pos[2]:12.10f}' for sym, pos in zip(symm_symbols, symm_positions))
    output.append('ATOMIC_POSITIONS crystal\n' + atoms_string)
    if 'k_points' in options_dict:
        output.append(f"K_POINTS {options_dict['k_points']['mode']}\n{options_dict['k_points']['input']}")

    qe_options['system']['nat'] = symm_positions.shape[0]
    qe_options['system']['ntyp'] = len(unique_symbols)

    lines = []
    for section, sec_vals in qe_options.items():
        if len(sec_vals) == 0:
            continue
        lines.append(f'&{section}')
        for inner_key, inner_val in sec_vals.items():
            if inner_val is not None:
                lines.append(qe_entry_string(inner_key, inner_val))
        lines.append('/')
    output.insert(0, '\n'.join(lines))
    return '\n\n'.join(output) + '\n\n'

def qe_pp_file(options_dict):
    qe_options = {
        'inputpp': {
            'prefix': 'scf',
            'outdir': './scratch',
            'plot_num': 21
        },
        'plot': {
            'iflag': 3,
            'output_format': 6,
            'fileout': 'density.cube'
        }
    }
    if 'control' in options_dict and 'prefix' in options_dict['control']:
        qe_options['inputpp']['prefix'] = options_dict['control']['prefix']
    if 'control' in options_dict and 'prefix' in options_dict['control']:
        qe_options['inputpp']['prefix'] = options_dict['control']['prefix']
    if 'core_electrons' in options_dict:
        # we have precalculated core electrons -> FT(core) has been done separately
        qe_options['inputpp']['plot_num'] = 17
    lines = []
    for section, sec_vals in qe_options.items():
        if len(sec_vals) == 0:
            continue
        lines.append(f'&{section}')
        for inner_key, inner_val in sec_vals.items():
            if inner_val is not None:
                lines.append(qe_entry_string(inner_key, inner_val))
        lines.append('/')
    return '\n'.join(lines) + '\n\n'
    
def qe_density(symm_symbols, symm_positions, cell_mat_m, options_dict):
    with open('pw.in', 'w') as fo:
        fo.write(qe_pw_file(symm_symbols, symm_positions, cell_mat_m, options_dict))
    #time.sleep(1)
    #with open('pw.out', 'w') as fo:
    #    subprocess.call(['pw.x', '-i', 'pw.in', '-o', 'pw.out'], stdout=fo, stderr=subprocess.DEVNULL, shell=True)
    subprocess.call(['pw.x -i pw.in'], stderr=subprocess.DEVNULL, shell=True)
    with open('pp.in', 'w') as fo:
        fo.write(qe_pp_file(options_dict))
    with open('pp.out', 'w') as fo:
        subprocess.call(['pp.x', '-i', 'pp.in'], stdout=fo, stderr=subprocess.DEVNULL)
    density, _ = cubetools.read_cube('density.cube')
    element_list = list(mass_dict.keys())
    n_elec = sum([element_list.index(symb) + 1 for symb in symm_symbols])
    if 'core_density' in options_dict:
        n_elec -= sum([options_dict['core_density'][symb] for symb in symm_symbols])
        print(options_dict['core_density'])
    print(len(symm_symbols), n_elec)
    return density * n_elec / density.sum()

def qe_atomic_density(symm_symbols, symm_positions, cell_mat_m, options_dict):
    at_options_dict = deepcopy(options_dict)
    if 'electrons' not in at_options_dict:
        at_options_dict['electrons'] = {}
    if 'control' not in at_options_dict:
        at_options_dict['control'] = {}
    at_options_dict['control']['prefix'] = 'adensity'
    at_options_dict['electrons']['electron_maxstep'] = 0
    at_options_dict['electrons']['startingwfc'] = 'atomic'
    at_options_dict['electrons']['scf_must_converge'] = False
    return qe_density(symm_symbols, symm_positions, cell_mat_m, at_options_dict)



def calc_f0j(cell_mat_m, element_symbols, positions, index_vec_h, symm_mats_vecs, options_dict=None, restart=None, explicit_core=True, save=None):
    """
    Calculate the aspherical atomic form factors from a density grid in the python package gpaw
    for each reciprocal lattice vector present in index_vec_h.
    """
    assert options_dict is not None, 'there is no default options_dict for the qe_source'
    options_dict = deepcopy(options_dict)
    if 'average_symmequiv' in options_dict:
        average_symmequiv = options_dict['average_symmequiv']
        #print(f'average symmetry equivalents: {average_symmequiv}')
        del(options_dict['average_symmequiv'])
    else:
        average_symmequiv = False
    if 'skip_symm' in options_dict:
        assert len(options_dict['skip_symm']) == 0 or average_symmequiv, 'skip_symm does need average_symmequiv' 
        skip_symm = options_dict['skip_symm']
        del(options_dict['skip_symm'])
    else:
        skip_symm = {}
    if restart:
        if 'electrons' not in options_dict:
            options_dict['electrons'] = {}
        options_dict['electrons']['startingwfc'] = 'file'


    #assert not (not average_symmequiv and not do_not_move)
    symm_positions, symm_symbols, f0j_indexes, magmoms_symm = expand_symm_unique(element_symbols,
                                                                                 np.array(positions),
                                                                                 np.array(cell_mat_m),
                                                                                 (np.array(symm_mats_vecs[0]), np.array(symm_mats_vecs[1])),
                                                                                 skip_symm=skip_symm,
                                                                                 magmoms=None)

    density = qe_density(symm_symbols, symm_positions, cell_mat_m, options_dict)
    print('  calculated density')

    overall_hdensity = qe_atomic_density(symm_symbols, symm_positions, cell_mat_m, options_dict)
    assert -density.shape[0] // 2 < index_vec_h[:,0].min(), 'Your gridspacing is too large.'
    assert density.shape[0] // 2 > index_vec_h[:,0].max(), 'Your gridspacing is too large.'
    assert -density.shape[1] // 2 < index_vec_h[:,1].min(), 'Your gridspacing is too large.'
    assert density.shape[1] // 2 > index_vec_h[:,1].max(), 'Your gridspacing is too large.'
    assert -density.shape[2] // 2 < index_vec_h[:,2].min(), 'Your gridspacing is too large.'
    assert density.shape[2] // 2 > index_vec_h[:,2].max(), 'Your gridspacing is too large.'
    f0j = np.zeros((symm_mats_vecs[0].shape[0], positions.shape[0], index_vec_h.shape[0]), dtype=np.complex128)

    if average_symmequiv:
        h, k, l = np.meshgrid(*map(lambda n: np.fft.fftfreq(n, 1/n).astype(np.int64), density.shape), indexing='ij')
        for atom_index, symm_atom_indexes in enumerate(f0j_indexes):
            f0j_sum = np.zeros_like(h, dtype=np.complex128)
            for symm_matrix, symm_atom_index in zip(symm_mats_vecs[0], symm_atom_indexes):
                atomic_density = qe_atomic_density([symm_symbols[symm_atom_index]], symm_positions[None,symm_atom_index,:],cell_mat_m, options_dict)
                h_density = density * atomic_density/ overall_hdensity
                frac_position = symm_positions[symm_atom_index]
                h_rot, k_rot, l_rot = np.einsum('xy, y... -> x...', symm_matrix, np.array((h, k, l))).astype(np.int64)
                phase_to_zero = np.exp(-2j * np.pi * (frac_position[0] * h + frac_position[1] * k + frac_position[2] * l))
                f0j_sum += (np.fft.ifftn(h_density) * phase_to_zero * np.prod(h.shape))[h_rot, k_rot, l_rot]
            f0j_sum /= len(symm_atom_indexes)

            for symm_index, symm_matrix in enumerate(symm_mats_vecs[0]):
                h_rot, k_rot, l_rot = np.einsum('xy, zy -> zx', symm_matrix.T, index_vec_h).astype(np.int64).T
                f0j[symm_index, atom_index, :] = f0j_sum[h_rot, k_rot, l_rot]
    else:
        #TODO Is a discrete Fourier Transform just of the hkl we need possibly faster? Can we then interpolate the density to get even better factors?
        # This could also save memory, fft is O(NlogN) naive dft is probably N^2
        h_vec, k_vec, l_vec = index_vec_h.T
        already_known = {}
        for atom_index, symm_atom_indexes in enumerate(f0j_indexes):
            f0j_sum = np.zeros_like(density, dtype=np.complex128)
            for symm_index, (symm_matrix, symm_atom_index) in enumerate(zip(symm_mats_vecs[0], symm_atom_indexes)):
                if symm_atom_index in list(already_known.keys()):
                    equiv_symm_index, equiv_atom_index = already_known[symm_atom_index]
                    f0j[symm_index, atom_index, :] = f0j[equiv_symm_index, equiv_atom_index, :].copy()
                else:
                    atomic_density = qe_atomic_density([symm_symbols[symm_atom_index]], symm_positions[None,symm_atom_index,:],cell_mat_m, options_dict)
                    h_density = density * atomic_density/ overall_hdensity
                    frac_position = symm_positions[symm_atom_index]
                    phase_to_zero = np.exp(-2j * np.pi * (frac_position[0] * h_vec + frac_position[1] * k_vec + frac_position[2] * l_vec))
                    f0j[symm_index, atom_index, :] = ((np.fft.ifftn(h_density) * np.prod(density.shape))[h_vec, k_vec, l_vec] * phase_to_zero).copy()
                    already_known[symm_atom_index] = (symm_index, atom_index)
    return f0j


def f_core_from_spline(spline, g_k, k=13):
    r_max = spline.get_cutoff()
    r = np.zeros(2**k + 1)
    r[1:] = np.exp(-1 * np.linspace(1.25 * k, 0.0 , 2**k)) * r_max
    #r[0] = 0
    gr = r[None,:] * g_k[:,None]
    j0 = np.zeros_like(gr)
    j0[gr != 0] = np.sin(2 * np.pi * gr[gr != 0]) / (2 * np.pi * gr[gr != 0])
    j0[gr == 0] = 1
    y00_factor = 0.5 * np.pi**(-0.5)
    int_me = 4 * np.pi * r**2  * spline.map(r) * j0
    return simps(int_me, x=r) * y00_factor


def calculate_f0j_core(cell_mat_m, element_symbols, index_vec_h, options_dict):
    ang_per_bohr = 0.529177210903
    cell_mat_f = np.linalg.inv(cell_mat_m).T
    g_k3 = np.einsum('xy, zy -> zx', cell_mat_f, index_vec_h)
    g_k = np.linalg.norm(g_k3, axis=-1) * ang_per_bohr
    pseudo_folder = options_dict['control']['pseudo_dir']
    core_factors_element = {}
    core_electrons = {}
    for element_symbol, upf_name in options_dict['paw_files'].items():
        assert upf_name.lower().endswith('.upf'), 'Currently only .upf files are implemented for core description'
        print(  f'  calculating core density for {element_symbol} from {upf_name}')
        with open(os.path.join(pseudo_folder, upf_name)) as fo:
            content = fo.read()
        ae_nlcc_search = re.search(r'<PP_AE_NLCC>(.+)</PP_AE_NLCC>', content, flags=re.DOTALL)
        if ae_nlcc_search is not None:
            core = np.array([float(val) for val in ae_nlcc_search.group(1).strip().split()])
        else:
            raise ValueError('No all-electron core density (entry <PP_AE_NLCC>) found in upf_file')
        mesh_search = re.search(r'<PP_R>(.+)</PP_R>', content, flags=re.DOTALL)
        if mesh_search is not None:
            r = np.array([float(val) for val in mesh_search.group(1).strip().split()])
        else:
            raise ValueError('No entry <PP_R> found in upf_file')
        core_electrons[element_symbol] = simps(4 * np.pi * r**2 * core, r)

        #r[0] = 0
        gr = r[None,:] * g_k[:,None]
        j0 = np.zeros_like(gr)
        j0[gr != 0] = np.sin(2 * np.pi * gr[gr != 0]) / (2 * np.pi * gr[gr != 0])
        j0[gr == 0] = 1
        y00_factor = 0.5 * np.pi**(-0.5)
        int_me = 4 * np.pi * r**2  * core * j0
        core_factors_element[element_symbol] = simps(int_me, x=r) * y00_factor
    options_dict['core_electrons'] = core_electrons
    return np.array([core_factors_element[symbol] for symbol in element_symbols]), options_dict