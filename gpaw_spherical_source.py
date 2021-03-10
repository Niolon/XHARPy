import numpy as np
import pickle

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
from .xharpy import expand_symm_unique
from .grid.atomgrid import AtomGrid
from .grid.onedgrid import HortonLinear
from .grid.rtransform import PowerRTransform
from .grid.utils import get_cov_radii
from .real_spher_harm import ylm_func_dict
from . import rho


def calc_f0j(cell_mat_m, element_symbols, positions, index_vec_h, symm_mats_vecs, gpaw_dict=None, restart=None, save='gpaw.gpw', explicit_core=True):
    """
    Calculate the aspherical atomic form factors from a density grid in the python package gpaw
    for each reciprocal lattice vector present in index_vec_h.
    """
    if gpaw_dict is None:
        gpaw_dict = {'xc': 'PBE', 'txt': 'gpaw.txt', 'h': 0.15, 'setups': 'paw', 'mode': 'lcao', 'basis': 'dzp'}
    else:
        gpaw_dict = gpaw_dict.copy()
    if 'gridrefinement' in gpaw_dict:
        warnings.warn('gridrefinement in gpaw_dict this is not used in spherical mode')
        del(gpaw_dict['gridrefinement'])
    if 'average_symmequiv' in gpaw_dict:
        warnings.warn("'average_symmequiv' is not allowed in spherical mode")
        del(gpaw_dict['average_symmequiv'])
    if 'spherical_grid' in gpaw_dict:
        grid_name = gpaw_dict['spherical_grid']
        del(gpaw_dict['spherical_grid'])
    else:
        grid_name = 'fine'

    #assert not (not average_symmequiv and not do_not_move)
    symm_positions, symm_symbols, inv_indexes = expand_symm_unique(element_symbols,
                                                                   np.array(positions),
                                                                   np.array(cell_mat_m),
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

    if save is not None:
        try:
            calc.write(save, mode='all')
        except:
            print('  Could not save to file')

    print('  calculated density with energy', e1)

    with pkg_resources.open_binary('xharpylib.rho', 'spherical_rho.pic') as fo:
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

    with pkg_resources.open_text('xharpylib.horton2-grids', grid_to_file[grid_name]) as fo:
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
    f0j = np.zeros((symm_mats_r.shape[0], inv_indexes.shape[1], index_vec_h.shape[0]), dtype=np.complex128)
    vec_s = np.einsum('xy, zy -> zx', np.linalg.inv(cell_mat_m / Bohr).T, index_vec_h)
    vec_s_symm = np.einsum('kxy, zx -> kzy', symm_mats_r, vec_s)

    xxx, yyy, zzz = np.meshgrid(np.arange(-10, 11, 1), np.arange(-10, 11, 1), np.arange(-10, 11, 1))
    supercell_base = np.array((np.ravel(xxx), np.ravel(yyy), np.ravel(zzz)))

    for z_atom_index, grid_atom_index in enumerate(inv_indexes[0]):
        setup_at = calc.setups[grid_atom_index]

        spline_at = spline_dict[setup_at.symbol]
        grid_vals = grid_dict[setup_at.Z]
        tr = PowerRTransform(grid_vals['lowlim'], grid_vals['highlim'])
        r_grid = tr.transform_1d_grid(HortonLinear(grid_vals['r_points']))
        center = atoms.get_positions()[grid_atom_index] / Bohr
        sp_grid = AtomGrid(r_grid, size=grid_vals['shells'], center=center)
        print(f'  Integrating atom {z_atom_index + 1}/{len(inv_indexes[0])}, n(Points): {sp_grid.points.shape[0]}, r(max): {grid_vals["highlim"]:6.4f} Ang')
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