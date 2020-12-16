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


class HirshfeldDensity(RealSpaceDensity):
    """Density as sum of atomic densities."""

    def __init__(self, calculator, log=None):
        self.calculator = calculator
        dens = calculator.density
        RealSpaceDensity.__init__(self, dens.gd, dens.finegd,
                                  dens.nspins, collinear=True, charge=0.0,
                                  stencil=dens.stencil,
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
