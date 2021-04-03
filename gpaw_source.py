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

import ase
from ase import Atoms
from ase.spacegroup import crystal

from scipy.interpolate import interp1d
from scipy.integrate import simps
import gpaw
import warnings
from .xharpy import expand_symm_unique


class HirshfeldDensity(RealSpaceDensity):
    """Density as sum of atomic densities."""

    def __init__(self, calculator, log=None):
        self.calculator = calculator
        dens = calculator.density
        try:
            RealSpaceDensity.__init__(self, dens.gd, dens.finegd,
                                    dens.nspins, collinear=True, charge=0.0,
                                    stencil=dens.stencil, 
                                    redistributor=dens.redistributor)
        except:
            RealSpaceDensity.__init__(self, dens.gd, dens.finegd,
                                    dens.nspins, collinear=True, charge=0.0,
                                    stencil=2, 
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
                        self.calculator.wfs.world)

        # initialize
        self.initialize(setups,
                        self.calculator.timer,
                        np.zeros((len(atoms), 3)), False)
        self.set_mixer(None)
        rank_a = self.gd.get_ranks_from_positions(spos_ac)
        self.set_positions(spos_ac, AtomPartition(self.gd.comm, rank_a))
        basis_functions = BasisFunctions(self.gd,
                                         [setup.phit_j
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



def calc_f0j(cell_mat_m, element_symbols, positions, index_vec_h, symm_mats_vecs, gpaw_dict=None, restart=None, save='gpaw.gpw', explicit_core=True):
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
    if 'skip_symm' in gpaw_dict:
        assert len(gpaw_dict['skip_symm']) == 0 or average_symmequiv, 'skip_symm does need average_symmequiv' 
        skip_symm = gpaw_dict['skip_symm']
        del(gpaw_dict['skip_symm'])
    else:
        skip_symm = {}

    #assert not (not average_symmequiv and not do_not_move)
    symm_positions, symm_symbols, f0j_indexes = expand_symm_unique(element_symbols,
                                                                   np.array(positions),
                                                                   np.array(cell_mat_m),
                                                                   (np.array(symm_mats_vecs[0]), np.array(symm_mats_vecs[1])),
                                                                   skip_symm=skip_symm)
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
                atoms, calc = gpaw.restart(restart, txt=gpaw_dict['txt'], xc=gpaw_dict['xc'])
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

    density = np.array(calc.get_all_electron_density(gridrefinement=gridrefinement, skip_core=explicit_core))
    if explicit_core:
        n_elec = sum([setup.Z for setup in calc.setups]) - sum(setup.Nc for setup in calc.density.setups)
    else:
        n_elec = sum([setup.Z for setup in calc.setups])
    density *= n_elec / density.sum()
    if save is not None:
        try:
            calc.write(save, mode='all')
        except:
            print('  Could not save to file')

    print('  calculated density with energy', e1)

    partitioning = HirshfeldPartitioning(calc)
    partitioning.initialize()
    overall_hdensity = partitioning.hdensity.get_density(list(range(symm_positions.shape[0])), gridrefinement=gridrefinement, skip_core=explicit_core)[0]
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
                h_density = density * partitioning.hdensity.get_density([symm_atom_index], gridrefinement=gridrefinement, skip_core=explicit_core)[0] / overall_hdensity
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
                    h_density = density * partitioning.hdensity.get_density([symm_atom_index], gridrefinement=gridrefinement, skip_core=explicit_core)[0] / overall_hdensity
                    frac_position = symm_positions[symm_atom_index]
                    phase_to_zero = np.exp(-2j * np.pi * (frac_position[0] * h_vec + frac_position[1] * k_vec + frac_position[2] * l_vec))
                    f0j[symm_index, atom_index, :] = ((np.fft.ifftn(h_density) * np.prod(density.shape))[h_vec, k_vec, l_vec] * phase_to_zero).copy()
                    already_known[symm_atom_index] = (symm_index, atom_index)
    del(calc)
    del(atoms)
    del(partitioning)
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
            print(f'  calculating the core structure factor by spline for {name}')
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
            print(f'  calculating the core structure factor for {name}')
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