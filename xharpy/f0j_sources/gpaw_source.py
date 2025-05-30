"""This module provides the necessary functions for calculating atomic form
factors using the GPAW library in single-core mode on a rectangular grid for the
valence density and a spherical grid for the core densities.
"""

from typing import Any, Dict, List, Tuple, Union
import numpy as np

from ase.units import Bohr
from gpaw.density import RealSpaceDensity
from gpaw.lfc import BasisFunctions
from gpaw.setup import Setups
from gpaw.xc import XC
from gpaw.utilities.tools import coordinates
from gpaw.utilities.partition import AtomPartition
from gpaw.mpi import world
from gpaw.io.logger import GPAWLogger

from ase.spacegroup import crystal

from scipy.interpolate import interp1d
from scipy.integrate import simpson
import gpaw
import warnings
from ..conversion import expand_symm_unique
from ..structure.construct import construct_values
from ..structure.common import AtomInstructions


class HirshfeldDensity(RealSpaceDensity):
    """Density as sum of atomic densities."""

    def __init__(self, calculator, log=None):
        self.calculator = calculator
        dens = calculator.density
        if hasattr(dens, 'stencil'):
            stencil = dens.stencil
        else:
            stencil = 2

        RealSpaceDensity.__init__(self, dens.gd, dens.finegd,
                                dens.nspins, collinear=True, charge=0.0,
                                stencil=stencil,
                                redistributor=dens.redistributor)

        self.log = GPAWLogger(world=world)
        if log is None:
            self.log.fd = None
        else:
            self.log.fd = log

    def set_positions(self, spos_ac, atom_partition):
        """HirshfeldDensity builds a hack density object to calculate
        all electron density
        of atoms. This methods overrides the parallel distribution of
        atomic density matrices
        in density.py"""
        self.atom_partition = atom_partition
        self.atomdist = self.redistributor.get_atom_distributions(spos_ac)
        self.nct.set_positions(spos_ac)
        self.ghat.set_positions(spos_ac)
        self.mixer.reset()
        # self.nt_sG = None
        self.nt_sg = None
        self.nt_g = None
        self.rhot_g = None
        self.Q_aL = None
        self.nct_G = self.gd.zeros()
        self.nct.add(self.nct_G, 1.0 / self.nspins)

    def get_density(self, atom_indices=None, gridrefinement=2, skip_core=False):
        """Get sum of atomic densities from the given atom list.

        Parameters
        ----------
        atom_indices : list_like
            All atoms are taken if the list is not given.
        gridrefinement : 1, 2, 4
            Gridrefinement given to get_all_electron_density

        Returns
        -------
        type
             spin summed density, grid_descriptor
        """

        all_atoms = self.calculator.get_atoms()
        if atom_indices is None:
            atom_indices = range(len(all_atoms))

        # select atoms
        atoms = self.calculator.get_atoms()[atom_indices]
        spos_ac = atoms.get_scaled_positions()
        Z_a = atoms.get_atomic_numbers()

        par = self.calculator.parameters
        setups = Setups(Z_a, par.setups, par.basis,
                        XC(par.xc),
                        world=self.calculator.wfs.world)

        # initialize
        self.initialize(setups,
                        self.calculator.timer,
                        np.zeros((len(atoms), 3)), False)
        self.set_mixer(None)
        rank_a = self.gd.get_ranks_from_positions(spos_ac)
        self.set_positions(spos_ac, AtomPartition(self.gd.comm, rank_a))
        if all(hasattr(setup, 'phit_j') for setup in self.setups):
            basis_functions = BasisFunctions(self.gd,
                                            [setup.phit_j
                                            for setup in self.setups],
                                            cut=True)
        else:
            basis_functions = BasisFunctions(self.gd,
                                            [setup.basis_functions_J
                                            for setup in self.setups],
                                            cut=True)
        basis_functions.set_positions(spos_ac)
        self.initialize_from_atomic_densities(basis_functions)

        aed_sg, gd = self.get_all_electron_density(atoms,
                                                   gridrefinement,
                                                   skip_core=skip_core)
        return aed_sg.sum(axis=0), gd


class HirshfeldPartitioning:
    """Partion space according to the Hirshfeld method.

    After: F. L. Hirshfeld Theoret. Chim.Acta 44 (1977) 129-138
    """

    def __init__(self, calculator, density_cutoff=1.e-12):
        self.calculator = calculator
        self.density_cutoff = density_cutoff

    def initialize(self):
        self.atoms = self.calculator.get_atoms()
        self.hdensity = HirshfeldDensity(self.calculator)
        density_g, gd = self.hdensity.get_density()
        self.invweight_g = 0. * density_g
        density_ok = np.where(density_g > self.density_cutoff)
        self.invweight_g[density_ok] = 1.0 / density_g[density_ok]

        den_sg, gd = self.calculator.density.get_all_electron_density(
            self.atoms)
        assert(gd == self.calculator.density.finegd)
        self.den_g = den_sg.sum(axis=0)

    def get_calculator(self):
        return self.calculator

    def get_effective_volume_ratios(self):
        """Return the list of effective volume to free volume ratios."""
        self.initialize()
        kptband_comm = self.calculator.comms['D']
        ratios = []
        for a, atom in enumerate(self.atoms):
            ratios.append(self.get_effective_volume_ratio(a))

        ratios = np.array(ratios)
        kptband_comm.broadcast(ratios, 0)
        return ratios

    def get_effective_volume_ratio(self, atom_index):
        """Effective volume to free volume ratio.

        After: Tkatchenko and Scheffler PRL 102 (2009) 073005, eq. (7)
        """
        finegd = self.calculator.density.finegd
        denfree_g, gd = self.hdensity.get_density([atom_index])
        assert(gd == finegd)

        # the atoms r^3 grid
        position = self.atoms[atom_index].position / Bohr
        r_vg, r2_g = coordinates(finegd, origin=position)
        r3_g = r2_g * np.sqrt(r2_g)

        weight_g = denfree_g * self.invweight_g

        nom = finegd.integrate(r3_g * self.den_g * weight_g)
        denom = finegd.integrate(r3_g * denfree_g)

        return nom / denom

    def get_weight(self, atom_index):
        denfree_g, gd = self.hdensity.get_density([atom_index])
        weight_g = denfree_g * self.invweight_g
        return weight_g

    def get_charges(self, den_g=None):
        """Charge on the atom according to the Hirshfeld partitioning

        Can be applied to any density den_g.
        """
        self.initialize()
        finegd = self.calculator.density.finegd

        if den_g is None:
            den_sg, gd = self.calculator.density.get_all_electron_density(
                self.atoms)
            den_g = den_sg.sum(axis=0)
        assert(den_g.shape == tuple(finegd.n_c))

        charges = []
        for ia, atom in enumerate(self.atoms):
            weight_g = self.get_weight(ia)
#            charge = atom.number - finegd.integrate(weight_g * den_g)
            charges.append(atom.number - finegd.integrate(weight_g * den_g))
        return charges



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
    GPAW.

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
          - gridinterpolation (1, 2, 4): Using GPAWs interpolation this is the
            factor by which the grid from the wave function will be interpolated
            for the calculation of atomic form factors with FFT. This can be
            reduced if you run out of memory for this step. Allowed values are
            1, 2, and 4, by default 4
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
          - core_grid (Union[str, int]): Determines how the core grid is build
            on which the core density is evaluated 'rgd' will use the default
            grid from GPAW, an integer k will span a grid of 2**k + 1 points,
            where the first point is 0 and all other points are determined by
            exp(-ai) * r_max, where ai is a np linspace between 1.25 * k and 0,
            by default 'rgd'
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
        gridinterpolation = computation_dict['gridinterpolation']
        #print(f'gridinterpolation set to {gridinterpolation}')
        del(computation_dict['gridinterpolation'])
    else:
        gridinterpolation = 4

    if 'save_file' in computation_dict:
        if computation_dict['save_file'] == 'none':
            save = None
            restart = False
        else:
            save = computation_dict['save_file']
        del(computation_dict['save_file'])
    else:
        save = 'gpaw_result.gpw'
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
    if 'magmoms' in computation_dict:
        magmoms = computation_dict['magmoms']
        del(computation_dict['magmoms'])
    else:
        magmoms = None
    if 'core_grid' in computation_dict:
        del(computation_dict['core_grid'])

    element_symbols = [instr.element for instr in construction_instructions]

    positions, *_ = construct_values(
        parameters,
        construction_instructions,
        cell_mat_m
    )

    #assert not (not average_symmequiv and not do_not_move)
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
        atoms.calc = calc
        e1 = atoms.get_potential_energy()
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                atoms, calc = gpaw.restart(save, txt=computation_dict['txt'], xc=computation_dict['xc'])
                e1_0 = atoms.get_potential_energy()

                atoms.set_scaled_positions(symm_positions % 1)
                e1 = atoms.get_potential_energy()
                e_change = abs(e1_0 - e1) > 1e-20
        except:
            print('  failed to load the density from previous calculation. Starting from scratch')
            atoms = crystal(symbols=symm_symbols,
                            basis=symm_positions % 1,
                            cell=cell_mat_m.T,
                            magmoms=magmoms_symm)
            calc = gpaw.GPAW(**computation_dict)
            atoms.calc = calc
            e1 = atoms.get_potential_energy()

    e1 = atoms.get_potential_energy()

    density = np.array(calc.get_all_electron_density(gridrefinement=gridinterpolation, skip_core=explicit_core))
    if explicit_core:
        n_elec = sum([setup.Z for setup in calc.setups]) - sum(setup.Nc for setup in calc.density.setups)
    else:
        n_elec = sum([setup.Z for setup in calc.setups])
    density *= n_elec / density.sum()
    if save is not None and e_change:
        try:
            calc.write(save, mode='all')
        except:
            print('  Could not save to file')

    print('  calculated density with energy', e1)

    partitioning = HirshfeldPartitioning(calc)
    partitioning.initialize()
    overall_hdensity = partitioning.hdensity.get_density(list(range(symm_positions.shape[0])), gridrefinement=gridinterpolation, skip_core=explicit_core)[0]
    assert -density.shape[0] // 2 < index_vec_h[:,0].min(), 'Your gridspacing is too large or wrongly read value for h in hkl.'
    assert density.shape[0] // 2 > index_vec_h[:,0].max(), 'Your gridspacing is too large or wrongly read value for h in hkl.'
    assert -density.shape[1] // 2 < index_vec_h[:,1].min(), 'Your gridspacing is too large or wrongly read value for k in hkl.'
    assert density.shape[1] // 2 > index_vec_h[:,1].max(), 'Your gridspacing is too large or wrongly read value for k in hkl.'
    assert -density.shape[2] // 2 < index_vec_h[:,2].min(), 'Your gridspacing is too large or wrongly read value for l in hkl.'
    assert density.shape[2] // 2 > index_vec_h[:,2].max(), 'Your gridspacing is too large or wrongly read value for l in hkl.'
    f0j = np.zeros((symm_mats_vecs[0].shape[0], positions.shape[0], index_vec_h.shape[0]), dtype=np.complex128)

    if symm_equiv == 'averaged':
        h, k, l = np.meshgrid(*map(lambda n: np.fft.fftfreq(n, 1/n).astype(np.int64), density.shape), indexing='ij')
        for atom_index, symm_atom_indexes in enumerate(f0j_indexes):
            f0j_sum = np.zeros_like(h, dtype=np.complex128)
            for symm_matrix, symm_atom_index in zip(symm_mats_vecs[0], symm_atom_indexes):
                h_density = density * partitioning.hdensity.get_density([symm_atom_index], gridrefinement=gridinterpolation, skip_core=explicit_core)[0] / overall_hdensity
                frac_position = symm_positions[symm_atom_index]
                hkl_all = np.vstack((h[None,:],k[None,:],l[None,:]))
                h_rot, k_rot, l_rot = np.einsum('x..., xy -> y...', hkl_all, symm_matrix).astype(np.int64)
                phase_to_zero = np.exp(-2j * np.pi * (frac_position[0] * h + frac_position[1] * k + frac_position[2] * l))
                f0j_symm1 = np.fft.ifftn(h_density) * phase_to_zero * np.prod(h.shape)
                f0j_sum += f0j_symm1[h_rot, k_rot, l_rot]
            f0j_sum /= len(symm_atom_indexes)

            for symm_index, symm_matrix in enumerate(symm_mats_vecs[0]):
                h_rot, k_rot, l_rot = np.einsum('xy, zy -> zx', symm_matrix.T, index_vec_h).astype(np.int64).T
                f0j[symm_index, atom_index, :] = f0j_sum[h_rot, k_rot, l_rot]
    elif symm_equiv == 'once':
        h, k, l = np.meshgrid(*map(lambda n: np.fft.fftfreq(n, 1/n).astype(np.int64), density.shape), indexing='ij')
        for atom_index, (symm_atom_index, *_) in enumerate(f0j_indexes):
            h_density = density * partitioning.hdensity.get_density([symm_atom_index], gridrefinement=gridinterpolation, skip_core=explicit_core)[0] / overall_hdensity
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
                    h_density = density * partitioning.hdensity.get_density([symm_atom_index], gridrefinement=gridinterpolation, skip_core=explicit_core)[0] / overall_hdensity
                    frac_position = symm_positions[symm_atom_index]
                    phase_to_zero = np.exp(-2j * np.pi * (frac_position[0] * h_vec + frac_position[1] * k_vec + frac_position[2] * l_vec))
                    f0j[symm_index, atom_index, :] = ((np.fft.ifftn(h_density) * np.prod(density.shape))[h_vec, k_vec, l_vec] * phase_to_zero).copy()
                    already_known[symm_atom_index] = (symm_index, atom_index)
    del(calc)
    del(atoms)
    del(partitioning)
    return f0j


def f_core_from_spline(
    spline: Any,
    g_k: np.ndarray,
    grid: Union[int, np.ndarray]= 13
) -> np.ndarray:
    """Calculate the spherical atomic form factor from a core density spline

    Parameters
    ----------
    spline : Any
        GPAW spline containing the core density for expansion. Anything else
        with a map and get_cutoff function should work too. The unit of length
        needs to be identical to the reciprocal of the unit used in g_k.
    g_k : np.ndarray
        reciprocal distances to origin for all. Reciprocal unit of length
        needs to be identical to the one used in the spline
    k : int, optional
        determines the number of distance points used for the evaluation
        as 2**k + 1. The first point is always zero, the other points are
        determined by np.exp(-1 * np.linspace(1.25 * k, 0.0 , 2**k)) * r_max,
        by default 13

    Returns
    -------
    f0j_core: np.ndarray
        calculated core atomic form factors for the reciprocal distances given
        in g_k
    """
    if type(grid) is int:
        r_max = spline.get_cutoff()
        r = np.zeros(2**grid + 1)
        r[1:] = np.exp(-1 * np.linspace(1.25 * grid, 0.0 , 2**grid)) * r_max
        #r[0] = 0
    if type(grid) is np.ndarray:
        r = grid
    gr = r[None,:] * g_k[:,None]
    j0 = np.zeros_like(gr)
    j0[gr != 0] = np.sin(2 * np.pi * gr[gr != 0]) / (2 * np.pi * gr[gr != 0])
    j0[gr == 0] = 1
    y00_factor = 0.5 * np.pi**(-0.5)
    int_me = 4 * np.pi * r**2  * spline.map(r) * j0
    return simpson(int_me, x=r) * y00_factor


def calc_f0j_core(
    cell_mat_m: np.ndarray,
    construction_instructions: List[AtomInstructions],
    parameters: np.ndarray,
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
        contains options for the calculation. The custom options will be ignored
        and everything else is passed on to GPAW for initialisation. The only
        option that makes a difference here is which setups are used. (Need to
        be same as in calc_f0j)

    Returns
    -------
    f0j_core: np.ndarray
        size (N, H) array of atomic core form factors calculated separately
    """

    element_symbols = [instr.element for instr in construction_instructions]

    positions, *_ = construct_values(
        parameters,
        construction_instructions,
        cell_mat_m
    )

    core_grid = computation_dict.get('core_grid', 'rgd')

    computation_dict = computation_dict.copy()
    non_gpaw_keys = [
        'gridinterpolation',
        'symm_equiv',
        'skip_symm',
        'magmoms',
        'save_file',
        'core_grid'
    ]
    for key in non_gpaw_keys:
        if key in computation_dict:
            del computation_dict[key]

    symm_positions, symm_symbols, *_ = expand_symm_unique(element_symbols,
                                                          positions,
                                                          cell_mat_m,
                                                          symm_mats_vecs)
    atoms = crystal(symbols=symm_symbols,
                    basis=symm_positions % 1,
                    cell=cell_mat_m.T)
    calc = gpaw.GPAW(**computation_dict)
    atoms.calc = calc
    calc.initialize(atoms)
    cell_inv = np.linalg.inv(atoms.cell.T).T
    g_k3 = np.einsum('xy, zy -> zx', cell_inv, index_vec_h)
    g_ks = np.linalg.norm(g_k3, axis=-1)
    splines = {setup.symbol: (setup.get_partial_waves()[2], setup.rgd.r_g) for setup in calc.density.setups}

    f0j_core = {}
    n_steps = 100
    n_per_step = 50

    for name, (nc, rgd) in list(splines.items()):
        if name in list(f0j_core.keys()):
            continue
        #if name == 'H':
        #    f0j_core[name] = np.zeros_like(g_ks)
        if core_grid == 'rgd':
            print(f'  calculating the core structure factor for {name}')
            f0j_core[name] = f_core_from_spline(nc, g_ks * Bohr, grid=rgd)
        elif len(g_ks) > n_steps * n_per_step:
            print(f'  calculating the core structure factor by spline for {name}')
            g_max = g_ks.max() * Bohr + 0.1
            #x_inv = np.linspace(-0.5, g_max, n_steps * n_per_step)
            k = np.log(n_steps * n_per_step)
            x_inv = np.exp(-1 * np.linspace(1.25 * k, 0.0, n_steps * n_per_step)) * g_max
            x_inv[0] = 0
            f0j = np.zeros(n_steps * n_per_step)
            for index in range(n_steps):
               f0j[index * n_per_step:(index + 1) * n_per_step] = f_core_from_spline(nc, x_inv[index * n_per_step:(index + 1) * n_per_step], grid=core_grid)
            f0j_core[name] = interp1d(x_inv, f0j, kind='cubic')(g_ks * Bohr)
        else:
            print(f'  calculating the core structure factor for {name}')
            f0j = np.zeros(len(g_ks))
            for index in range(n_per_step, len(g_ks) + n_per_step, n_per_step):
                start_index = index - n_per_step
                if index < len(g_ks):
                    end_index = index
                else:
                    end_index = len(g_ks)
                f0j[start_index:end_index] = f_core_from_spline(nc, g_ks[start_index:end_index] * Bohr, grid=core_grid)
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
    derived from theoretically calculated densities
  - Density calculation was done with ASE/GPAW using the
    following settings
{value_strings}
  - Afterwards density was interpolated on a rectangular grid and partitioned
    according to the Hirshfeld scheme, using GPAWs build-in routines.
  - Atomic form factors were calculated using FFT from the numpy package"""
    return addition