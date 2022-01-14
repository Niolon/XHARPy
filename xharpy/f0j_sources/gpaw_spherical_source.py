import numpy as np
import pickle
from typing import List, Tuple, Dict, Any

from ase.units import Bohr
from gpaw.spherical_harmonics import Y
from gpaw.utilities import unpack2

import ase
from ase import Atoms
from ase.spacegroup import crystal

from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.special import spherical_jn
import gpaw
import warnings
import importlib.resources as pkg_resources
from ..core import expand_symm_unique, construct_values, AtomInstructions
from .grid.atomgrid import AtomGrid
from .grid.onedgrid import HortonLinear
from .grid.rtransform import PowerRTransform
from .grid.utils import get_cov_radii
from .real_spher_harm import ylm_func_dict


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
    GPAW and a spherical grid expansion onto a grid as implemented in HORTON. 

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
        will use and exclude the following options from the dictionary and pass
        the rest onto the GPAW calculator without further checks.

          - save_file (str): Path to the file that is used for saving and 
            loading DFT results, by default 'gpaw_result.gpw'

          - spherical_grid (str): Can be used to select a grid. Possible options
            are: coarse, medium, fine, veryfine, ultrafine and insane, by
            by default 'fine'

          - skip_symm (Dict[int, List[int]]): Can used to prevent the
            expansion of the atom(s) with the index(es) given as dictionary keys
            as given in the construction_instructions with the symmetry
            operations of the indexes given in the list, which correspond to the
            indexes in the symm_mats_vecs object. This has proven to be
            successful for the calculation of atoms disordered on special 
            positions. Can only be used with average_symmequiv, by default {} 

          - magmoms (np.ndarray): Experimental: starting values for magnetic
            moments of atoms. These will be expanded to atoms in the unit cell 
            by just applying the same magnetic moment to all symmetry equivalent
            atoms. This is probably too simplistic and will fail.

        For the allowed options of the GPAW calculator consult: 
        https://wiki.fysik.dtu.dk/gpaw/documentation/basic.html
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
    computation_dict = computation_dict.copy()

    if 'gridinterpolation' in computation_dict:
        warnings.warn('gridinterpolation in computation_dict. This is not used in spherical mode')
        del(computation_dict['gridinterpolation'])

    if 'save_file' in computation_dict:
        if computation_dict['save_file'] == 'none':
            save = None
            restart = False
        else:
            save = computation_dict['save_file']
        del(computation_dict['save_file'])
    else:
        save = 'gpaw_result.gpw'

    if 'average_symmequiv' in computation_dict:
        warnings.warn("'average_symmequiv' is not allowed in spherical mode")
        del(computation_dict['average_symmequiv'])

    if 'spherical_grid' in computation_dict:
        grid_name = computation_dict['spherical_grid']
        del(computation_dict['spherical_grid'])
    else:
        grid_name = 'fine'
    if 'skip_symm' in computation_dict:
        assert len(computation_dict['skip_symm']) == 0, 'skip_symm not allowed in this mode' 
        skip_symm = computation_dict['skip_symm']
        del(computation_dict['skip_symm'])
    else:
        skip_symm = {}
    if 'magmoms' in computation_dict:
        magmoms = computation_dict['magmoms']
        del(computation_dict['magmoms'])
    else:
        magmoms = None

    element_symbols = [instr.element for instr in construction_instructions]

    positions, *_ = construct_values(
        parameters,
        construction_instructions,
        cell_mat_m
    )

    symm_positions, symm_symbols, f0j_indexes, magmoms_symm = expand_symm_unique(
        element_symbols,
        np.array(positions),
        np.array(cell_mat_m),
        (np.array(symm_mats_vecs[0]), np.array(symm_mats_vecs[1])),
        skip_symm=skip_symm,
        magmoms=magmoms
    )
    e_change = True
    if not restart:
        atoms = crystal(symbols=symm_symbols,
                        basis=symm_positions % 1,
                        cell=cell_mat_m.T,
                        magmoms=magmoms_symm)
        calc = gpaw.GPAW(**computation_dict)
        atoms.set_calculator(calc)
        e1 = atoms.get_potential_energy()
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                atoms, calc = gpaw.restart(save, txt=computation_dict['txt'])
                e1_0 = atoms.get_potential_energy()

                atoms.set_scaled_positions(symm_positions % 1)
                e1 = atoms.get_potential_energy()
                e_change = abs(e1_0 - e1) > 1e-20
        except:
            print('  failed to load the density from previous calculation. Starting from scratch')
            atoms = crystal(symbols=symm_symbols,
                            basis=symm_positions % 1,
                            cell=cell_mat_m.T)
            calc = gpaw.GPAW(**computation_dict)
            atoms.set_calculator(calc)
            e1 = atoms.get_potential_energy()

    e1 = atoms.get_potential_energy()

    if save is not None and e_change:
        try:
            calc.write(save, mode='all')
        except:
            print('  Could not save to file')

    print('  calculated density with energy', e1)

    with pkg_resources.open_binary('xharpy.f0j_sources.rho', 'spherical_rho.pic') as fo:
        atomic_dict = pickle.load(fo)
        
    spline_dict = {}
    #for symbol in set([setup.symbol for setup in calc.density.setups]):
    for setup in calc.density.setups:
        if setup.symbol in spline_dict:
            continue
        atom_grid_1d, rho = atomic_dict[setup.symbol]
        if explicit_core:
            _, _, nc, *_ = setup.get_partial_waves()
            rho -= nc.map(atom_grid_1d)
        atom_grid_1d[0] = 0
        spline_dict[setup.symbol] = interp1d(atom_grid_1d, rho, 'cubic', fill_value=(np.nan, 0.0), bounds_error=False)
    symm_mats_r, _ = symm_mats_vecs

    grid_to_file = {
        'coarse':    'tv-13.7-3.txt',
        'medium':    'tv-13.7-4.txt',
        'fine':      'tv-13.7-5.txt',
        'veryfine':  'tv-13.7-6.txt',
        'ultrafine': 'tv-13.7-7.txt',
        'insane':    'tv-13.7-8.txt'
    }

    with pkg_resources.open_text('xharpy.f0j_sources.horton2-grids', grid_to_file[grid_name]) as fo:
        lines = [line.strip() for line in fo.readlines() if len(line.strip()) > 1 and line[0] != '#']
    grid_dict = {}
    for line1, line2, line3 in zip(lines[::3], lines[1::3], lines[2::3]):
        z = int(line1.strip().split()[0])
        _, lower_str, upper_str, npoints_str = line2.strip().split()
        shells = [int(val) for val in line3.split()]
        grid_dict[z] = {
            'lowlim': float(lower_str),
            'highlim': float(upper_str),
            'r_points': int(npoints_str),
            'shells': shells
        }
    #print(symm_mats_r, inv_indexes, index_vec_h)
    f0j = np.zeros((symm_mats_r.shape[0], positions.shape[0], index_vec_h.shape[0]), dtype=np.complex128)
    vec_s = np.einsum('xy, zy -> zx', np.linalg.inv(cell_mat_m / Bohr).T, index_vec_h)
    vec_s_symm = np.einsum('kxy, zx -> kzy', symm_mats_r, vec_s)

    xxx, yyy, zzz = np.meshgrid(np.arange(-10, 11, 1), np.arange(-10, 11, 1), np.arange(-10, 11, 1))
    supercell_base = np.array((np.ravel(xxx), np.ravel(yyy), np.ravel(zzz)))

    for z_atom_index, grid_atom_index in enumerate([index[0] for index in f0j_indexes]):
        setup_at = calc.setups[grid_atom_index]

        spline_at = spline_dict[setup_at.symbol]
        grid_vals = grid_dict[setup_at.Z]
        tr = PowerRTransform(grid_vals['lowlim'], grid_vals['highlim'])
        r_grid = tr.transform_1d_grid(HortonLinear(grid_vals['r_points']))
        center = atoms.get_positions()[grid_atom_index] / Bohr
        sp_grid = AtomGrid(r_grid, size=grid_vals['shells'], center=center)
        print(f'  Integrating atom {z_atom_index + 1}/{len(f0j_indexes)}, n(Points): {sp_grid.points.shape[0]}, r(max): {grid_vals["highlim"]:6.4f} Ang')
        grid = sp_grid.points.T
        dens_mats = [calc.wfs.calculate_density_matrix(kpt.f_n, kpt.C_nM) for kpt in calc.wfs.kpt_u]
        atomic_wfns_gd = np.zeros((dens_mats[0].shape[0], *grid.shape[1:]))
        density_atom = np.zeros(grid.shape[1:])#, dtype=np.complex128)
        collect_har = np.zeros_like(grid[0])
        wfs_index = 0
        D_asp = calc.density.D_asp

        for atom_index, (position_sc, setup) in enumerate(zip(atoms.get_scaled_positions(), calc.density.setups)):

            phis, phits, nc, *_ = setup.get_partial_waves()
            basis_funcs = setup.basis.tosplines()
            center_distances = np.linalg.norm(np.einsum('xy, yk -> xk', cell_mat_m / Bohr, position_sc[:, None] + supercell_base[:, :]) - center[:, None], axis=0) 
            supercell = supercell_base[:,center_distances < grid_vals['highlim'] + setup.basis.rgd.r_g[-1]]
            position_supercell = np.einsum('xy, yzk -> xzk', cell_mat_m / Bohr, position_sc[:, None, None] + supercell[:, :, None]) - grid[:, None, :]
            distances = np.linalg.norm(position_supercell, axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                direction_cosines = position_supercell / distances[None, :,]
                direction_cosines[np.isnan(direction_cosines)] = 0 # if atom_position is on grid
            maxl = max([max([phi.l for phi in phis]), max([phit.l for phit in phits]), max([basis_func.l for basis_func in basis_funcs])])
            Ys = [Y(index, *direction_cosines) * (-distances)**np.floor(np.sqrt(index)) for index in range(0, (maxl + 1)**2)]
            n_projector = sum([2 * phi.l + 1 for phi in phis])
            projector_wave = np.zeros((n_projector, *distances.shape[1:]))
            projector_wave_t = np.zeros((n_projector, *distances.shape[1:]))
            projector_index = 0
            for phi, phit in zip(phis, phits):
                assert phi.get_cutoff() == phit.get_cutoff()
                phi_map = np.zeros_like(distances)
                phit_map = np.zeros_like(distances)
                condition = distances < phi.get_cutoff()
                phi_map[condition] = phi.map(distances[condition])
                phit_map[condition] = phit.map(distances[condition])
                for y_index in range(phi.l**2, (phi.l + 1)**2):
                    inner = phi_map.copy()
                    inner[condition] *= Ys[y_index][condition]
                    projector_wave[projector_index] = np.sum(inner, axis=0)
                    inner = phit_map.copy()
                    inner[condition] *= Ys[y_index][condition]
                    projector_wave_t[projector_index] = np.sum(inner, axis=0)
                    projector_index += 1
            if not explicit_core:
                density_atom += np.sum(nc.map(distances) * Ys[0], axis=0)

            for D_p in D_asp[atom_index]:
                density_atom += np.einsum('x..., y..., xy -> ...', projector_wave, projector_wave, unpack2(D_p))
                density_atom -= np.einsum('x..., y..., xy -> ...', projector_wave_t, projector_wave_t, unpack2(D_p))

            for basis_func in basis_funcs:
                condition = distances < basis_func.get_cutoff()
                value = np.zeros_like(distances)
                value[condition] = basis_func.map(distances[condition])
                for y_index in range(basis_func.l**2, (basis_func.l + 1)**2):
                    inner = value.copy()
                    inner[condition] *= Ys[y_index][condition]
                    atomic_wfns_gd[wfs_index] = np.sum(inner, axis=0)
                    wfs_index += 1
            collect_har += np.sum(spline_dict[setup.symbol](distances), axis=0)

        for dens_mat in dens_mats:
            density_atom += np.einsum('x..., y..., xy -> ...', atomic_wfns_gd, atomic_wfns_gd, dens_mat)

        distances = np.linalg.norm(grid.T - sp_grid.center, axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            direction_cosines = (grid.T - sp_grid.center) / distances[:, None]
            direction_cosines[np.isnan(direction_cosines)] = 0

        h_density = density_atom * spline_at(distances) / collect_har
        if explicit_core:
            print(f'  Integrated Hirshfeld Charge: {setup_at.Z - setup_at.Nc - np.real(sp_grid.integrate(h_density)):6.4f}')
        else:   
            print(f'  Integrated Hirshfeld Charge: {setup_at.Z - np.real(sp_grid.integrate(h_density)):6.4f}')
        f0j[:, z_atom_index, :] = np.array([[sp_grid.integrate(h_density * np.exp(2j * np.pi * np.einsum('x, zx -> z', vec, sp_grid.points - sp_grid.center))) for vec in vec_s] for vec_s in vec_s_symm])
    return f0j


def f_core_from_spline(spline, g_k, k=13):
    r_max = spline.get_cutoff()
    r = np.zeros(2**k + 1)
    r[1:] = np.exp(-1 * np.linspace(1.25 * k, 0.0 , 2**k)) * r_max
    r[0] = 0
    gr = r[None,:] * g_k[:,None]
    j0 = np.zeros_like(gr)
    j0[gr != 0] = np.sin(2 * np.pi * gr[gr != 0]) / (2 * np.pi * gr[gr != 0])
    j0[gr == 0] = 1
    y00_factor = 0.5 * np.pi**(-0.5)
    int_me = 4 * np.pi * r**2  * spline.map(r) * j0
    return simps(int_me, x=r) * y00_factor


def calc_f0j_core(
    cell_mat_m: np.ndarray,
    element_symbols: List[str],
    positions: np.ndarray,
    index_vec_h: np.ndarray,
    symm_mats_vecs: np.ndarray,
    computation_dict: Dict[str, Any]
) -> np.ndarray:
    """Calculate the core atomic form factors on an exponential spherical grid.
    Up to 5000 reflections every reflection will be calculated explicitely. 
    Above that a spline will be generated from 5000 points on an exponential
    grid. The spline is then used to calculate the individual atomic core form
    factor values.

    Parameters
    ----------
    cell_mat_m : np.ndarray
        size (3, 3) array with the unit cell vectors as row vectors
    element_symbols : List[str]
        element symbols (i.e. 'Na') for all the atoms within the asymmetric unit
    positions : np.ndarray
        atomic positions in fractional coordinates for all the atoms within
        the asymmetric unit
    index_vec_h : np.ndarray
        size (H) vector containing Miller indicees of the measured reflections
    symm_mats_vecs : Tuple[np.ndarray, np.ndarray]
        size (K, 3, 3) array of symmetry  matrices and (K, 3) array of
        translation vectors for all symmetry elements in the unit cell
    computation_dict : Dict[str, Any]
        contains options for the calculation. The custom options will be ignored
        and everything else is passed on to GPAW for initialisation. The only
        option that makes a difference here is which setups are used. (Need to
        be same as in calc_f0j)

    Returns
    -------
    f0j_core: np.ndarray
        size (N, H) array of atomic core form factors calculated separately
    """
    warnings.warn('The Hirshfeld weights in this mode are not calculated without core density so there might be some mismatch')
    computation_dict = computation_dict.copy()
    non_gpaw_keys = [
        'gridinterpolation',
        'average_symmequiv',
        'skip_symm',
        'magmoms'
    ]
    for key in non_gpaw_keys:
        if key in computation_dict:
            del computation_dict[key]

    symm_positions, symm_symbols, _ = expand_symm_unique(element_symbols,
                                                                   positions,
                                                                   cell_mat_m,
                                                                   symm_mats_vecs)
    atoms = crystal(symbols=symm_symbols,
                    basis=symm_positions % 1,
                    cell=cell_mat_m.T)
    calc = gpaw.GPAW(**computation_dict)
    atoms.set_calculator(calc)
    calc.initialize(atoms)
    cell_inv = np.linalg.inv(atoms.cell.T).T
    g_k3 = np.einsum('xy, zy -> zx', cell_inv, index_vec_h)
    g_ks = np.linalg.norm(g_k3, axis=-1)
    splines = {setup.symbol: setup.get_partial_waves()[:4] for setup in calc.density.setups}

    f0j_core = {}
    n_steps = 100
    n_per_step = 50

    for name, (_, _, nc, _) in list(splines.items()):
        if name in list(f0j_core.keys()):
            continue
        #if name == 'H':
        #    f0j_core[name] = np.zeros_like(g_ks)
        if len(g_ks) > n_steps * n_per_step:
            print(f'  Calculating the core structure factor by spline for {name}')
            g_max = g_ks.max() * Bohr + 0.1
            #x_inv = np.linspace(-0.5, g_max, n_steps * n_per_step)
            k = np.log(n_steps * n_per_step)
            x_inv = np.exp(-1 * np.linspace(1.25 * k, 0.0, n_steps * n_per_step)) * g_max 
            x_inv[0] = 0
            f0j = np.zeros(n_steps * n_per_step)
            for index in range(n_steps):
               f0j[index * n_per_step:(index + 1) * n_per_step] = f_core_from_spline(nc, x_inv[index * n_per_step:(index + 1) * n_per_step], k=19) 
            f0j_core[name] = interp1d(x_inv, f0j, kind='cubic')(g_ks * Bohr)
        else:
            print(f'  Calculating the core structure factor for {name}')
            f0j = np.zeros(len(g_ks))
            for index in range(n_per_step, len(g_ks) + n_per_step, n_per_step):
                start_index = index - n_per_step
                if index < len(g_ks):
                    end_index = index
                else:
                    end_index = len(g_ks)
                f0j[start_index:end_index] = f_core_from_spline(nc, g_ks[start_index:end_index] * Bohr, k=19)
            f0j_core[name] = f0j
    return np.array([f0j_core[symbol] for symbol in element_symbols])


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
    derived from theoretically calculated atomic densities
  - Density calculation was done with ASE/GPAW using the
    following settings
{value_strings}
  - Afterwards density was interpolated on a spherical grid from the horton
    package
  - Atomic form factors were calculated using discrete fourier transform"""
    return addition