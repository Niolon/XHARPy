"""This module provides the necessary function to do use Quantum Espresso
as source for the atomic form factors. Still very experimental but works in
principle."""

import shutil
from typing import Union, List, Dict, Any, Tuple
import numpy as np
from copy import deepcopy
import subprocess
import os
import datetime
import re
from . import cubetools
import functools
from multiprocessing import Pool


#from scipy.interpolate import interp1d
from scipy.integrate import simpson
#from scipy.ndimage import zoom
import warnings
from ..conversion import expand_symm_unique
from ..structure.construct import construct_values
from ..structure.common import AtomInstructions

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

ang_per_bohr = 0.529177210903


def qe_entry_string(
    name: str,
    value: Union[str, float, int, bool],
    string_sign: bool = True
) -> str:
    """Creates a formatted string for output in a quantum-espresso input file

    Parameters
    ----------
    name : str
        Name of the option
    value : Union[str, float, int, bool]
        The value of the option
    string_sign : bool, optional
        If the value is a string this value determines, whether the entry,
        will have '' as an indicator of the type, by default True

    Returns
    -------
    str
        Formatted string

    Raises
    ------
    NotImplementedError
        The type of value is currently not implemented
    """
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

def qe_pw_file(
    symm_symbols: List[str],
    symm_positions: np.ndarray,
    cell_mat_m: np.ndarray,
    computation_dict: Dict[str, Any]
) -> str:
    """Creates an input file for pw.x for the density calculation

    Parameters
    ----------
    symm_symbols : List[str]
        Atom Type indicators for the symmetry expanded atoms in the unit cell
    symm_positions : np.ndarray
        atomic positions in fractional coordinates for the symmetry expanded
        atoms in the unit cell.
    cell_mat_m : np.ndarray
        size (3, 3) array with the unit cell vectors as row vectors, only used
        if ibrav != 0
    computation_dict : Dict[str, Any]
        Dictionary with the calculation options, see calc_f0j function for
        options

    Returns
    -------
    pw_file_string : str
        formatted output file as a string
    """
    qe_options = {
        'control': {
            'prefix': 'scf',
            'calculation': 'scf',
            'restart_mode': 'from_scratch',
            'pseudo_dir': './pseudo/',
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
    for section, secval in computation_dict.items():
        if section in ('paw_files', 'core_electrons', 'k_points',
                       'mpicores', 'density_format', 'pw_in_file',
                       'pw_out_file', 'pp_in_file', 'pp_out_file',
                       'non_convergence', 'pw_executable', 'pp_executable',
                       'windows', 'omp_num_threads'):
            # these are either given in tables in QE or not used here
            continue
        if section in qe_options:
            qe_options[section].update(secval)
        else:
            qe_options[section] = secval
    output = []
    unique_symbols = set(symm_symbols)
    assert 'paw_files' in computation_dict, "'paw_files' entry in computation_dict is missing"
    lines = []
    for symbol in unique_symbols:
        assert symbol in computation_dict['paw_files'], f"No paw file (e.g. .UPF) given for {symbol} in computation_dict['paw_files']"
        pp = computation_dict['paw_files'][symbol]
        mass = mass_dict[symbol]
        lines.append(f'{symbol} {mass:8.3f}  {pp}')
    pp_string = '\n'.join(lines)
    output.append('ATOMIC_SPECIES\n' + pp_string)
    if qe_options['system']['ibrav'] == 0:
        cell_mat_str = '\n'.join(f'{line[0]:9.6f} {line[1]:9.6f} {line[2]:9.6f}' for line in cell_mat_m.T)
        output.append('CELL_PARAMETERS angstrom\n' + cell_mat_str)
    atoms_string = '\n'.join(f'{sym} {pos[0]:12.10f} {pos[1]:12.10f} {pos[2]:12.10f}' for sym, pos in zip(symm_symbols, symm_positions))
    output.append('ATOMIC_POSITIONS crystal\n' + atoms_string)
    if 'k_points' in computation_dict:
        output.append(f"K_POINTS {computation_dict['k_points']['mode']}\n{computation_dict['k_points']['input']}")

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

def qe_pp_file(computation_dict: Dict[str, Any]) -> str:
    """Creates an input file for pp.x

    Parameters
    ----------
    computation_dict : Dict[Any]
        Dictionary with the calculation options, see calc_f0j function for
        options

    Returns
    -------
    pp_file_string : str
        a pp.x input file as string
    """
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
    if 'control' in computation_dict and 'prefix' in computation_dict['control']:
        qe_options['inputpp']['prefix'] = computation_dict['control']['prefix']
    if 'core_electrons' in computation_dict:
        # we have precalculated core electrons -> FT(core) has been done separately
        qe_options['inputpp']['plot_num'] = 17

    density_format = computation_dict.get('density_format', 'xsf')
    if density_format == 'xsf':
        qe_options['plot']['output_format'] = 5
        qe_options['plot']['fileout'] = 'density.xsf'
    elif density_format == 'cube':
        pass # is used as the default above
    else:
        raise NotImplementedError('unknown density format allowed options are: xsf, cube')


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

def read_xsf_density(filename: str) -> np.ndarray:
    """Reads the density from an xsf file.

    Parameters
    ----------
    filename : str
        Path to the xsf file

    Returns
    -------
    density : np.ndarray
        density as a numpy array. Will cut off the repeated points at the edges
        that are included in the file.
    """

    with open(filename, 'r') as fo:
        content = fo.read()

    start_expr = r'BEGIN_DATAGRID_3D_\w+\n'
    points_expr = r'\s*(\d+)\s+(\d+)\s+(\d+)\s*\n'
    origin_expr = r'\s*([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s*\n'
    vec_expr = r'\s*([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s*\n'
    data_expr = r'([\n\d\.\+\-Ee\s]+)'
    end_expr = r'END_DATAGRID_3D'

    expr = start_expr + points_expr + origin_expr + 3* vec_expr + data_expr + end_expr

    result = re.search(expr, content)
    finds = result.groups()

    n_points = np.array([int(val) for val in finds[:3]])
    #origin = np.array([float(val) for val in finds[3:6]])
    #vecs = np.array([float(val) for val in finds[6:15]]).reshape(3,3)
    data = np.array([float(val) for val in finds[15].strip().split()])
    density = np.reshape(data, n_points, order='F')[:-1, :-1, :-1]
    return density.copy()

def qe_density(
    symm_symbols: List[str],
    symm_positions: np.ndarray,
    cell_mat_m: np.ndarray,
    computation_dict: Dict[str, Any],
    cwd=os.getcwd()
) -> np.ndarray:
    """
    Performs the wavefunction calculation with quantum espresso, generates the
    density cube file with or without the core density and finally loads the
    density with cubetools.

    Parameters
    ----------
    symm_symbols : List[str]
        Atom Type indicators for the symmetry expanded atoms in the unit cell
    symm_positions : np.ndarray
        atomic positions in fractional coordinates for the symmetry expanded
        atoms in the unit cell.
    cell_mat_m : np.ndarray
        size (3, 3) array with the unit cell vectors as row vectors, only used
        if ibrav != 0
    computation_dict : Dict[str, Any]
        Dictionary with the calculation options, see calc_f0j function for
        options

    Returns
    -------
    density : np.ndarray
        Numpy array containing the density. The overall sum of the array is
        normalised to the number of electrons.
    """
    is_atomic = 'electrons' in computation_dict and computation_dict['electrons'].get('electron_maxstep', 1) == 0
    if is_atomic:
        in_pw = 'pw_atomic.in'
        in_pp = 'pp_atomic.in'
        pw_executable = computation_dict.get('pw_executable', 'pw.x')
        pp_executable = computation_dict.get('pp_executable', 'pp.x')
        out_pw = '/dev/null'
        out_pp = '/dev/null'
    else:
        in_pw = computation_dict.get('pw_in_file', 'pw.in')
        in_pp = computation_dict.get('pp_in_file', 'pp.in')
        pw_executable = computation_dict.get('pw_executable', 'pw.x')
        pp_executable = computation_dict.get('pp_executable', 'pp.x')
        out_pw = computation_dict.get('pw_out_file', 'pw.out')
        out_pp = computation_dict.get('pw_out_file', 'pp.out')

    for filename in (in_pw, in_pp, out_pw, out_pp):
        if filename == '/dev/null':
            continue
        if os.path.exists(filename):
            os.remove(filename)

    with open(os.path.join(cwd, in_pw), 'w') as fo:
        fo.write(qe_pw_file(symm_symbols, symm_positions, cell_mat_m, computation_dict))
    with open(os.path.join(cwd, in_pp), 'w') as fo:
        fo.write(qe_pp_file(computation_dict))
    mpicores = computation_dict['mpicores']
    omp_num_threads = computation_dict['omp_num_threads']

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(omp_num_threads)
    env['I_MPI_PIN_DOMAIN'] = '1'

    if mpicores > 1:
        pw_executable = f'mpirun -np {mpicores} ' + pw_executable
        pp_executable = f'mpirun -np {mpicores} ' + pp_executable

    if computation_dict.get('windows', False):
        cwd = os.getcwd()
        in_pw_w = os.path.join(cwd, in_pw)
        if out_pw == '/dev/null':
            out_pw_w = 'NUL'
        else:
            out_pw_w = os.path.join(cwd, out_pw)
        in_pp_w = os.path.join(cwd, in_pp)
        if out_pp == '/dev/null':
            out_pp_w = 'NUL'
        else:
            out_pp_w = os.path.join(cwd, out_pp)
        res = subprocess.call(rf'{pw_executable} -i {in_pw_w} > {out_pw_w}', shell=True)
        assert res in (0, 2), 'Calculation did not finish'
        if os.path.exists(os.path.join(cwd, 'CRASH')):
            raise FileExistsError('CRASH exists so the pw.x calculation failed')
        res = subprocess.call(rf'{pp_executable} -i {in_pp_w} > {out_pp_w}', shell=True)
        assert res == 0, 'Density generation failed'
        if os.path.exists(os.path.join(cwd, 'CRASH')):
            raise FileExistsError('CRASH exists so the pp.x calculation failed')
    else:
        subprocess.call(
            [f'{pw_executable} -x OMP_NUM_THREADS={omp_num_threads} -i {in_pw} > {out_pw}'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True,
            cwd=cwd,
            env=env
        )
        assert not os.path.exists(os.path.join(cwd, 'CRASH')), 'pw.x crashed during runtime'
        subprocess.call(
            [f'{pp_executable} -x OMP_NUM_THREADS={omp_num_threads} -i {in_pp} > {out_pp}'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True,
            cwd=cwd,
            env=env
        )
    assert not os.path.exists(os.path.join(cwd, 'CRASH')), 'pp.x crashed during runtime'

    if not is_atomic:
        with open(out_pw) as fo:
            pw_content = fo.read()
        is_converged = re.search(r'convergence has been achieved in\s+\d+\s+iterations', pw_content) is not None
        if is_converged:
            final_energy = re.findall(r'!\s+total energy\s+=\s+(\-?[\d]*\.\d+) Ry', pw_content)[-1]
            print(f'  convergence has been achieved with energy of {final_energy} Ry')
        else:
            non_conv = computation_dict.get('non_convergence', 'exception')
            if non_conv == 'exception':
                raise ValueError('Quantum Espresso SCF calculation has not converged.')
            elif non_conv == 'warning':
                warnings.warn('Quantum Espresso SCF calculation has not converged. Your HAR result is almost certainly not reliable!')
            elif non_conv == 'print':
                print('  Quantum Espresso SCF calculation has not converged. Your HAR result is almost certainly not reliable!')
            else:
                raise NotImplementedError('Treatment of non-convergence is not known, use exception, warning or print')

    density_format = computation_dict.get('density_format', 'xsf')
    if  density_format == 'xsf':
        density = read_xsf_density(os.path.join(cwd, 'density.xsf'))
        os.remove(os.path.join(cwd, 'density.xsf'))
    elif density_format == 'cube':
        density, _ = cubetools.read_cube(os.path.join(cwd, 'density.cube'))
        os.remove(os.path.join(cwd, 'density.cube'))
    else:
        raise NotImplementedError('unknown density format allowed options are: xsf, cube')

    element_list = list(mass_dict.keys())

    n_elec = sum([element_list.index(symb) + 1 for symb in symm_symbols])
    if 'core_electrons' in computation_dict:
        n_elec -= sum([computation_dict['core_electrons'][symb] for symb in symm_symbols])
    return density * n_elec / density.sum()

def qe_atomic_density(
    symm_symbols: List[str],
    symm_positions: np.ndarray,
    cell_mat_m: np.ndarray,
    computation_dict: Dict[str, Any],
    shape: Tuple[int] = None,
    cwd=os.getcwd()
) -> np.ndarray:
    """
    Generates the atomic function needed for Hirshfeld partitioning by
    setting the quantum espresso options to 0 calculation steps and
    initialisation to atomic. Subsequently, generates the density cube file
    with or without the core density and finally loads the
    density with cubetools.

    Parameters
    ----------
    symm_symbols : List[str]
        Atom Type indicators for the evaluated atom(s)
    symm_positions : np.ndarray
        atomic positions in fractional coordinates for the evaluated atom(s)
    cell_mat_m : np.ndarray
        size (3, 3) array with the unit cell vectors as row vectors, only used
        if ibrav != 0
    computation_dict : Dict[str, Any]
        Dictionary with the calculation options, see calc_f0j function for
        options
    shape : Tuple[int]
        tuple containing the shape of the desired atomic density. This circumvents
        an inconsistency of QE on how large the densities arrays are depending
        on some parameters in the file.

    Returns
    -------
    atomic_density : np.ndarray
        Numpy array containing the atomic_density. The overall sum of the array
        is normalised to the number of electrons.
    """
    at_computation_dict = deepcopy(computation_dict)
    if 'electrons' not in at_computation_dict:
        at_computation_dict['electrons'] = {}
    if 'control' not in at_computation_dict:
        at_computation_dict['control'] = {}

    at_computation_dict['mpicores'] = 1

    if shape is not None:
        if 'system' not in at_computation_dict:
            at_computation_dict['system'] = {}
        at_computation_dict['system']['nr1'] = shape[0]
        at_computation_dict['system']['nr2'] = shape[1]
        at_computation_dict['system']['nr3'] = shape[2]
    at_computation_dict['control']['prefix'] = 'adensity'
    at_computation_dict['electrons']['electron_maxstep'] = 0
    at_computation_dict['electrons']['startingwfc'] = 'atomic'
    at_computation_dict['electrons']['startingpot'] = 'atomic'
    at_computation_dict['electrons']['scf_must_converge'] = False
    return qe_density(symm_symbols, symm_positions, cell_mat_m, at_computation_dict, cwd)

def single_core_function(
    atom_index,
    symm_atom_index,
    symm_mats_vecs,
    index_vec_h,
    computation_dict,
    symm_symbols,
    symm_positions,
    cell_mat_m,
    density,
    overall_hdensity
):
    h, k, l = np.meshgrid(*map(lambda n: np.fft.fftfreq(n, 1/n).astype(np.int64), density.shape), indexing='ij')

    f0j_atom = np.zeros((symm_mats_vecs[0].shape[0], index_vec_h.shape[0]), dtype=np.complex128)
    sc_computation_dict = deepcopy(computation_dict)
    sc_computation_dict['control']['pseudo_dir'] = os.path.abspath(sc_computation_dict['control']['pseudo_dir'])
    sc_computation_dict['omp_num_threads'] = 1
    # create individual folders for each atom
    atom_dir = f'atomic_{atom_index}'
    if os.path.exists(atom_dir):
        shutil.rmtree(atom_dir)
    os.mkdir(atom_dir)
    atomic_density = qe_atomic_density(
        [symm_symbols[symm_atom_index]],
        symm_positions[None,symm_atom_index,:],
        cell_mat_m,
        sc_computation_dict,
        density.shape,
        atom_dir
    )
    shutil.rmtree(atom_dir)

    h_density = density * atomic_density / overall_hdensity
    frac_position = symm_positions[symm_atom_index]
    phase_to_zero = np.exp(-2j * np.pi * (frac_position[0] * h + frac_position[1] * k + frac_position[2] * l))
    f0j_symm1 = np.fft.ifftn(h_density) * phase_to_zero * np.prod(h.shape)
    for symm_index, symm_matrix in enumerate(symm_mats_vecs[0]):
        h_rot, k_rot, l_rot = np.einsum('zx, xy -> zy', index_vec_h, symm_matrix).T.astype(np.int64)
        f0j_atom[symm_index, :] = f0j_symm1[h_rot, k_rot, l_rot]
    return f0j_atom

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
    Quantum espresso.

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
        contains options for the atomic form factor calculation. The function
        will use and exclude the following options from the dictionary and write
        the rest into the quantum-espresso pw.x output file without further
        check

          - mpicores (Union[str, int]): The number of cores used for the pw.x
            and pp.x calculation in Quantum Espresso, 'auto' will mpiexec let
            select this option. However sometimes it has proven faster to
            choose a lower number of cores manually. This is not the only option
            for parallelisation: setting mpicores to 1 might still use non-MPI
            means of multi-core calculations.
          - symm_equiv (str): The atomic form factors of symmetry equivalent
            atoms can be calculated individually for each atom ('individually')
            or they can be calculated once for each atom in the asymmetric unit
            and expanded to the other atoms ('once'), finally they can be
            averaged between symmetry equivalent atoms and expanded afterwards
            ('averaged'). Once should be sufficient for most structures and
            saves time. Try one of the other options if you suspect problems,
            by default 'once'
          - skip_symm (Dict[int, List[int]]): Can used to prevent the
            expansion of the atom(s) with the index(es) given as dictionary keys
            as given in the construction_instructions with the symmetry
            operations of the indexes given in the list, which correspond to the
            indexes in the symm_mats_vecs object. This has proven to be
            successful for the calculation of atoms disordered on special
            positions. Can not be used with if symm_equiv is 'individually',
            by default {}
          - pw_in_file (str): Filename for the input file of the pw.x scf
            calculation, by default pw.in
          - pp_in_file (str): Filename for the input file of pp.x, by default
            pp.in
          - pw_out_file (str): Filename for the output file of the pw.x scf
            calculation, by default pw.out
          - pp_out_file (str): Filename for the output file of pp.x,
            by default pp.out
          - pw_executable (str): Filename or path to the executable of pw,
            by default pw.x
          - pp_executable (str): Filename or path to the executable of pp,
            by default pp.x
          - windows (bool): If set to True, Windows PowerShell commands will
            be used to call Quantum Espresso, by default False
          - non_convergence (str): How to deal with non-convergence in SCF
            'exception' will stop the calculation with a ValueError, 'warning'
            will print out a warning module 'print' will only print the warning
            in the usual text, by default 'exception'

        K-points are organised into their own entry 'k_points' which is a dict
        'mode' is the selection mode, and 'input' is the output after the
        K_POINTS entry in the pw.x output file.

        The other options are organised as subdicts with the naming of the
        section in the pw.x input file in lowercase.
        For these options consult the pw.x file format documentation at:
        https://www.quantum-espresso.org/Doc/INPUT_PW.html

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

    assert computation_dict is not None, 'there is no default computation_dict for the qe_source'
    computation_dict = deepcopy(computation_dict)
    if 'symm_equiv' in computation_dict:
        symm_equiv = computation_dict['symm_equiv']
        if symm_equiv not in ('once', 'averaged', 'individually'):
            raise NotImplementedError('symm_equiv treatment must be once, averaged or individually')
        del(computation_dict['symm_equiv'])
    else:
        symm_equiv = 'once'
    if 'skip_symm' in computation_dict:
        assert len(computation_dict['skip_symm']) == 0 or symm_equiv in ('once', 'averaged'), 'skip_symm does need symm_equiv once or averaged'
        skip_symm = computation_dict['skip_symm']
        del(computation_dict['skip_symm'])
    else:
        skip_symm = {}

    mpicores = computation_dict.get('mpicores', 1)
    omp_threads = computation_dict.get('omp_num_threads', os.cpu_count())

    if mpicores == 'auto':
        mpicores = os.cpu_count()
        if mpicores is None:
            mpicores = 1
        elif omp_threads != 'auto':
            mpicores = mpicores // omp_threads
            if mpicores == 0:
                mpicores = 1
    if omp_threads == 'auto':
        omp_threads = os.cpu_count() // mpicores
        if omp_threads == 0:
            omp_threads = 1
    assert type(mpicores) is int, 'mpicores has to either "auto" or int'

    computation_dict['mpicores'] = mpicores
    computation_dict['omp_num_threads'] = omp_threads


    if restart:
        if 'electrons' not in computation_dict:
            computation_dict['electrons'] = {}
        computation_dict['electrons']['startingwfc'] = 'file'
        computation_dict['electrons']['startingpot'] = 'file'

    element_symbols = [instr.element for instr in construction_instructions]

    positions, *_ = construct_values(
        parameters,
        construction_instructions,
        cell_mat_m
    )

    symm_positions, symm_symbols, f0j_indexes, magmoms_symm = expand_symm_unique(element_symbols,
                                                                                 np.array(positions),
                                                                                 np.array(cell_mat_m),
                                                                                 (np.array(symm_mats_vecs[0]), np.array(symm_mats_vecs[1])),
                                                                                 skip_symm=skip_symm,
                                                                                 magmoms=None)
    print('  theoretical calculation started at ', datetime.datetime.now())
    density = qe_density(symm_symbols, symm_positions, cell_mat_m, computation_dict)
    #print('  calculated density, continuing with partitioning')
    print('  theoretical calculation ended at ', datetime.datetime.now())

    print('  partitioning started at ', datetime.datetime.now())
    overall_hdensity = qe_atomic_density(symm_symbols, symm_positions, cell_mat_m, computation_dict, density.shape)
    assert -density.shape[0] // 2 < index_vec_h[:,0].min(), 'Your gridspacing is too large or wrongly read value for h in hkl.'
    assert density.shape[0] // 2 > index_vec_h[:,0].max(), 'Your gridspacing is too large or wrongly read value for h in hkl.'
    assert -density.shape[1] // 2 < index_vec_h[:,1].min(), 'Your gridspacing is too large or wrongly read value for k in hkl.'
    assert density.shape[1] // 2 > index_vec_h[:,1].max(), 'Your gridspacing is too large or wrongly read value for k in hkl.'
    assert -density.shape[2] // 2 < index_vec_h[:,2].min(), 'Your gridspacing is too large or wrongly read value for l in hkl.'
    assert density.shape[2] // 2 > index_vec_h[:,2].max(), 'Your gridspacing is too large or wrongly read value for l in hkl.'
    f0j = np.zeros((symm_mats_vecs[0].shape[0], positions.shape[0], index_vec_h.shape[0]), dtype=np.complex128)

    if symm_equiv == 'once' and mpicores > 1:

        single_core_function_red = functools.partial(
            single_core_function,
            symm_mats_vecs=symm_mats_vecs,
            index_vec_h=index_vec_h,
            computation_dict=computation_dict,
            symm_symbols=symm_symbols,
            symm_positions=symm_positions,
            cell_mat_m=cell_mat_m,
            density=density,
            overall_hdensity=overall_hdensity
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Pool(mpicores) as p:
                f0j_atoms = p.starmap(single_core_function_red, enumerate(indexes[0] for indexes in f0j_indexes))
            for atom_index, f0j_atom in enumerate(f0j_atoms):
                f0j[:, atom_index, :] = f0j_atom
    elif symm_equiv == 'averaged':
        h, k, l = np.meshgrid(*map(lambda n: np.fft.fftfreq(n, 1/n).astype(np.int64), density.shape), indexing='ij')
        for atom_index, symm_atom_indexes in enumerate(f0j_indexes):
            f0j_sum = np.zeros_like(h, dtype=np.complex128)
            for symm_matrix, symm_atom_index in zip(symm_mats_vecs[0], symm_atom_indexes):
                atomic_density = qe_atomic_density(
                    [symm_symbols[symm_atom_index]],
                    symm_positions[None,symm_atom_index,:],
                    cell_mat_m,
                    computation_dict,
                    density.shape
                )
                h_density = density * atomic_density/ overall_hdensity
                frac_position = symm_positions[symm_atom_index]
                h_rot, k_rot, l_rot = np.einsum('zx, xy -> zy', index_vec_h, symm_matrix).T.astype(np.int64)
                phase_to_zero = np.exp(-2j * np.pi * (frac_position[0] * h + frac_position[1] * k + frac_position[2] * l))
                f0j_sum += (np.fft.ifftn(h_density) * phase_to_zero * np.prod(h.shape))[h_rot, k_rot, l_rot]
            f0j_sum /= len(symm_atom_indexes)

            for symm_index, symm_matrix in enumerate(symm_mats_vecs[0]):
                h_rot, k_rot, l_rot = np.einsum('xy, zy -> zx', symm_matrix.T, index_vec_h).astype(np.int64).T
                f0j[symm_index, atom_index, :] = f0j_sum[h_rot, k_rot, l_rot]
    elif symm_equiv == 'once':
        h, k, l = np.meshgrid(*map(lambda n: np.fft.fftfreq(n, 1/n).astype(np.int64), density.shape), indexing='ij')
        for atom_index, (symm_atom_index, *_) in enumerate(f0j_indexes):
            atomic_density = qe_atomic_density(
                [symm_symbols[symm_atom_index]],
                symm_positions[None,symm_atom_index,:],
                cell_mat_m,
                computation_dict,
                density.shape
            )
            h_density = density * atomic_density/ overall_hdensity
            frac_position = symm_positions[symm_atom_index]
            phase_to_zero = np.exp(-2j * np.pi * (frac_position[0] * h + frac_position[1] * k + frac_position[2] * l))
            f0j_symm1 = np.fft.ifftn(h_density) * phase_to_zero * np.prod(h.shape)
            for symm_index, symm_matrix in enumerate(symm_mats_vecs[0]):
                h_rot, k_rot, l_rot = np.einsum('zx, xy -> zy', index_vec_h, symm_matrix).T.astype(np.int64)
                f0j[symm_index, atom_index, :] = f0j_symm1[h_rot, k_rot, l_rot]
    elif symm_equiv == 'individually':
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
                    atomic_density = qe_atomic_density(
                        [symm_symbols[symm_atom_index]],
                        symm_positions[None,symm_atom_index,:],
                        cell_mat_m,
                        computation_dict,
                        density.shape
                    )
                    h_density = density * atomic_density/ overall_hdensity
                    frac_position = symm_positions[symm_atom_index]
                    phase_to_zero = np.exp(-2j * np.pi * (frac_position[0] * h_vec + frac_position[1] * k_vec + frac_position[2] * l_vec))
                    f0j[symm_index, atom_index, :] = ((np.fft.ifftn(h_density) * np.prod(density.shape))[h_vec, k_vec, l_vec] * phase_to_zero).copy()
                    already_known[symm_atom_index] = (symm_index, atom_index)
    print('  partitioning ended at ', datetime.datetime.now())
    return f0j

def calc_f0j_core(
    cell_mat_m: np.ndarray,
    construction_instructions: List[AtomInstructions],
    parameters: np.ndarray,
    index_vec_h: np.ndarray,
    computation_dict: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reads the .upf files given in the computation_dict and fourier transforms
    the core charges on the grid given in that file. A direct space transform
    will be used to add the number of core_electrons to the returned
    computation_dict for correct normalisation of densities in the calc_f0j
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
    computation_dict : Dict[str, Any]
        contains options for the calculation. The custom options will be ignored
        and everything else is passed on to GPAW for initialisation. The only
        option that makes a difference here is which setups are used. (Need to
        be same as in calc_f0j)

    Returns
    -------
    f0j_core : np.ndarray,
        size (N, H) array of atomic core form factors calculated separately
    computation_dict: Dict[str, Any]
        original computation dict with added core electrons.

    Raises
    ------
    ValueError
        No core electron entry found
    ValueError
        No grid entry found
    """

    element_symbols = [instr.element for instr in construction_instructions]

    cell_mat_f = np.linalg.inv(cell_mat_m).T
    g_k3 = np.einsum('xy, zy -> zx', cell_mat_f, index_vec_h)
    g_k = np.linalg.norm(g_k3, axis=-1) * ang_per_bohr
    pseudo_folder = computation_dict['control']['pseudo_dir']
    core_factors_element = {}
    core_electrons = {}
    for element_symbol, upf_name in computation_dict['paw_files'].items():
        assert upf_name.lower().endswith('.upf'), 'Currently only .upf files are implemented for core description'
        print(  f'  calculating core density for {element_symbol} from {upf_name}')
        with open(os.path.join(pseudo_folder, upf_name)) as fo:
            content = fo.read()
        ae_nlcc_search = re.search(r'<PP_AE_NLCC(?:\s.*?>|>)(.+)</PP_AE_NLCC>', content, flags=re.DOTALL)
        if ae_nlcc_search is not None:
            core = np.array([float(val) for val in ae_nlcc_search.group(1).strip().split()])
        else:
            raise ValueError('No all-electron core density (entry <PP_AE_NLCC>) found in upf_file')
        mesh_search = re.search(r'<PP_R(?:\s.*?>|>)(.+)</PP_R>', content, flags=re.DOTALL)
        if mesh_search is not None:
            r = np.array([float(val) for val in mesh_search.group(1).strip().split()])
        else:
            raise ValueError('No entry <PP_R> found in upf_file')
        core_electrons[element_symbol] = simpson(4 * np.pi * r**2 * core, x=r)

        #r[0] = 0
        gr = r[None,:] * g_k[:,None]
        j0 = np.zeros_like(gr)
        j0[gr != 0] = np.sin(2 * np.pi * gr[gr != 0]) / (2 * np.pi * gr[gr != 0])
        j0[gr == 0] = 1
        int_me = 4 * np.pi * r**2  * core * j0
        core_factors_element[element_symbol] = simpson(int_me, x=r)
    computation_dict['core_electrons'] = core_electrons
    return np.array([core_factors_element[symbol] for symbol in element_symbols]), computation_dict


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
  - Density calculation was done with Quantum Espresso using the
    following settings
{value_strings}
  - Afterwards density was interpolated on a rectangular grid and partitioned
    according to the Hirshfeld scheme, using atomic densities from Quantum
    Espresso.
  - Atomic form factors were calculated using FFT from the numpy package"""
    return addition