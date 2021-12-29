import numpy as np
import os

from ase.units import Bohr
from gpaw.utilities import unpack2
from gpaw import restart
from itertools import product

from grid.atomgrid import AtomGrid
from grid.rtransform import HyperbolicRTransform
from grid.onedgrid import OneDGrid

import denspart
from grid.periodicgrid import PeriodicGrid
from denspart.mbis import MBISProModel
from denspart.vh import optimize_reduce_pro_model
import contextlib

import ase
from ase import Atoms
from ase.spacegroup import crystal

from scipy.interpolate import interp1d
from scipy.integrate import simps
import gpaw
import warnings
from .core import expand_symm_unique

from numpy.testing import assert_allclose
from scipy.interpolate import CubicSpline


def spherical_harmonics(work, lmax, solid=False, racah=None):
    """Recursive calculation of spherical harmonics.
    Parameters
    ----------
    work
        The input and output array. First three elements should contain x, y and z.
        After calling this function, the spherical harmonics are stored in Horton 2
        order: c10 c11 s11 c20 c21 s21 c22 s22 c30 c31 s31 c32 s32 c33 s33 ...
        (c stands for cosine-like, s for sine like, first ditit is l, second digit is m.)
    lmax
        Maximum angular momentum. The work array should have at least (lmax + 1)**2 - 1
        elements along the first dimension.
    solid
        When True, the real regular solid harmonics are computed instead of the normal
        spherical harmonics.
    racah
        Use Racah's normalization. The default is False for conventional spherical harmonics
        and True for solid harmonics. Setting this to False for solid harmonics will
        raise an error. When ``racah==True``, the L2 norm of the spherical harmonics is
        4 pi / (2 l + 1).
    """
    if racah is None:
        racah = solid
    if solid and not racah:
        raise ValueError(
            "Regular solid spherical harmonics always use racah normalization."
        )
    if work.shape[0] < (lmax + 1) ** 2 - 1:
        raise ValueError("Work array is too small for given lmax.")

    shape = work[0].shape
    z = work[0]
    x = work[1]
    y = work[2]

    r2 = x * x + y * y + z * z
    if not solid:
        r = np.sqrt(r2)
        mask = r > 0
        rmask = r[mask]
        z[mask] /= rmask
        x[mask] /= rmask
        y[mask] /= rmask
        r2[mask] = 1

    # temporary arrays to store PI(z,r) polynomials
    tmp_shape = (lmax + 1,) + shape
    pi_old = np.zeros(tmp_shape)
    pi_new = np.zeros(tmp_shape)
    a = np.zeros(tmp_shape)
    b = np.zeros(tmp_shape)

    # Initialize the temporary arrays
    pi_old[0] = 1
    pi_new[0] = z
    pi_new[1] = 1
    a[1] = x
    b[1] = y

    old_offset = 0  # first array index of the moments of the previous shell
    old_npure = 3  # number of moments in previous shell
    for l in range(2, lmax + 1):
        new_npure = old_npure + 2
        new_offset = old_offset + old_npure

        # Polynomials PI(z,r) for current l
        factor = 2 * l - 1
        for m in range(l - 1):
            tmp = pi_old[m].copy()
            pi_old[m] = pi_new[m]
            pi_new[m] = (z * factor * pi_old[m] - r2 * (l + m - 1) * tmp) / (l - m)

        pi_old[l - 1] = pi_new[l - 1]
        pi_new[l] = factor * pi_old[l - 1]
        pi_new[l - 1] = z * pi_new[l]

        # construct new polynomials A(x,y) and B(x,y)
        a[l] = x * a[l - 1] - y * b[l - 1]
        b[l] = x * b[l - 1] + y * a[l - 1]

        # construct solid harmonics
        work[new_offset] = pi_new[0]
        factor = np.sqrt(2)
        for m in range(1, l + 1):
            factor /= np.sqrt((l + m) * (l - m + 1))
            work[new_offset + 2 * m - 1] = factor * a[m] * pi_new[m]
            work[new_offset + 2 * m] = factor * b[m] * pi_new[m]
        old_npure = new_npure
        old_offset = new_offset

    if not (solid or racah):
        work /= 2 * np.sqrt(np.pi)
        begin = 0
        end = 3
        for l in range(1, lmax + 1):
            print(begin, end, 2 * l + 1)
            work[begin:end] *= np.sqrt(2 * l + 1)
            begin = end
            end += 2 * l + 3


def prepare_input(atoms, calc):
    """Prepare input for denspart from a GPAW run.
    Parameters
    ----------
    atoms
        A list of ASE atoms from a GPAW calculation.
    calc
        The GPAW calculator instance
    Returns
    -------
    input_data
        A dictionary with all input data for a partitioning.
    """
    # Get some basic system information.
    atnums = atoms.get_atomic_numbers()
    atcorenums = atoms.get_atomic_numbers()
    atcoords = atoms.get_positions(wrap=True) / Bohr
    cellvecs = calc.density.gd.cell_cv

    print("Loading uniform grid data")
    uniform_data = get_uniform_grid_data(calc, cellvecs, atnums)
    print("Loading setups & atomic density matrices")
    setups, atoms = get_atomic_grid_data(calc)
    print("Computing corrections in augmentation spheres")
    compute_augmentation_spheres(uniform_data, setups, atoms, atnums, atcoords)
    print("Computing uniform grid info")
    compute_uniform_points(uniform_data)
    print("Convert to denspart arrays")
    density = denspart_conventions(uniform_data, atoms)
    density.update(
        {
            "atcoords": atcoords,
            "atnums": atnums,
            "atcorenums": atcorenums,
            "cellvecs": cellvecs,
        }
    )

    print("Final check")
    print(
        "  total charge = {:10.3e}".format(
            density["atnums"].sum() - np.dot(density["weights"], density["density"])
        )
    )

    return density


def get_uniform_grid_data(calc, cellvecs, atnums):
    """Take the (pseudo) density on the the uniform grid. This is the easy part.
    Parameters
    ----------
    calc
        GPAW calculator instance.
    cellvecs
        3 x 3 array whose rows are cell vectors.
    atnums
        Array with atomic numbers.
    Returns
    -------
    uniform_data
        Dictionary with several items extracted from the GPAW calculator.
    """
    # Parameters that determine the sizes of most grids
    data = {}
    data["shape"] = calc.density.gd.N_c
    data["grid_vecs"] = calc.density.gd.h_cv
    data["nspins"] = calc.wfs.nspins
    for i in range(3):
        assert np.allclose(calc.density.gd.h_cv[i], cellvecs[i] / data["shape"][i])

    # Load (pseudo) density data on the uniform grid.
    if calc.wfs.nspins == 1:
        # Spin-paired case
        data["charge_corrections"] = calc.get_pseudo_density_corrections()
        data["pseudo_density"] = calc.get_pseudo_density() * (Bohr ** 3)
        # Conversion to atomic units is needed. (?)
        data["ae_density"] = calc.get_all_electron_density(gridrefinement=1) * (
            Bohr ** 3
        )
    else:
        # Spin-polarized case, convert to spin-sum and spin-difference densitities.
        corrections = calc.get_pseudo_density_corrections()
        data["charge_corrections"] = corrections[0] + corrections[1]
        data["spincharge_corrections"] = corrections[0] - corrections[1]

        density_pseudo_alpha = calc.get_pseudo_density(0) * (Bohr ** 3)
        density_pseudo_beta = calc.get_pseudo_density(1) * (Bohr ** 3)
        data["pseudo_density"] = density_pseudo_alpha + density_pseudo_beta
        data["pseudo_spindensity"] = density_pseudo_alpha - density_pseudo_beta

        # Conversion to atomic units is needed. (?)
        density_ae_alpha = calc.get_all_electron_density(spin=0, gridrefinement=1) * (
            Bohr ** 3
        )
        density_ae_beta = calc.get_all_electron_density(spin=1, gridrefinement=1) * (
            Bohr ** 3
        )
        data["ae_density"] = density_ae_alpha + density_ae_beta
        data["ae_spindensity"] = density_ae_alpha - density_ae_beta

    # Sanity checks
    assert (data["pseudo_density"].shape == data["shape"]).all()
    assert (data["ae_density"].shape == data["shape"]).all()
    # w = is the quadrature weight for the uniform grid.
    w = abs(np.linalg.det(data["grid_vecs"]))
    q_pseudo = data["pseudo_density"].sum() * w
    q_corr = data["charge_corrections"].sum()
    assert np.allclose(q_pseudo, -q_corr)

    if calc.wfs.nspins == 2:
        # some checks specific for spin-polarized results
        assert (data["pseudo_spindensity"].shape == data["shape"]).all()
        assert (data["ae_spindensity"].shape == data["shape"]).all()
        qspin_pseudo = data["pseudo_spindensity"].sum() * w
        qspin_corr = data["spincharge_corrections"].sum()
        assert np.allclose(qspin_pseudo, -qspin_corr)

    # We're assuming all systems in GPAW are neutral. In fact, this is not strictly True
    # in all cases. We may have to relax this a little.
    q_ae = data["ae_density"].sum() * w
    assert_allclose(q_ae, atnums.sum())

    return data


def get_atomic_grid_data(calc):
    """Load atomic setups and atomic wavefunctions from GPAW calculation.
    Parameters
    ----------
    calc
        GPAW calculator instance.
    Returns
    -------
    setups
        A dictionary with atomic setups used. Keys are atomic numbers and values are
        dictionaries with relevant data for later evaluation of the density corrections
        within the augmentation spheres.
    atoms
        A list with atomic wavefunction data. Contains dm and optional spindm.
    """
    setups = {}
    atoms = []

    for iatom, id_setup in enumerate(calc.density.setups.id_a):
        setup = calc.density.setups[iatom]

        if id_setup not in setups.keys():
            print("  Converting setup", id_setup)
            # We have not encountered it before, time to parse the new setup.
            setup_data = {}
            # Angular momenta of the shells of basis functions.
            setup_data["ls"] = setup.l_j
            order = get_horton2_order(setup_data["ls"])
            setup_data["order"] = order
            # Get the overlap matrix, mostly for debugging.
            setup_data["overlap"] = setup.dO_ii[order][:, order]

            # Dump spline basis functions for nc and nct.
            # These are the core density functions (projected and all-electron).
            dump_spline(setup_data, ("nc",), setup.data.nc_g, setup, 0)
            dump_spline(setup_data, ("nct",), setup.data.nct_g, setup, 0)
            # Dump splines basis for phi and phit.
            # These are the local atomic orbital basis functions (projected and all-electron).
            for iradial, phi_g in enumerate(setup.data.phi_jg):
                l = setup_data["ls"][iradial]
                dump_spline(setup_data, ("phi", iradial), phi_g, setup, l)
            for iradial, phit_g in enumerate(setup.data.phit_jg):
                l = setup_data["ls"][iradial]
                dump_spline(setup_data, ("phit", iradial), phit_g, setup, l)
            setups[id_setup] = setup_data
        else:
            # Reuse setup that was previously loaded and take the reordering of the
            # basis functions.
            order = setups[id_setup]["order"]

        atom_data = {}
        if calc.wfs.nspins == 1:
            atom_data["dm"] = unpack2(calc.density.D_asp.get(iatom)[0])[order][:, order]
        else:
            # spin-summed and spin-difference atomic density matrices.
            dma = unpack2(calc.density.D_asp.get(iatom)[0])[order][:, order]
            dmb = unpack2(calc.density.D_asp.get(iatom)[1])[order][:, order]
            atom_data["dm"] = dma + dmb
            atom_data["spindm"] = dma - dmb
        assert atom_data["dm"].shape == (setup.ni, setup.ni)
        atom_data["id_setup"] = id_setup

        atoms.append(atom_data)

    return setups, atoms


def get_horton2_order(ls):
    """Return a permutation of the basis functions to obtain HORTON 2 conventions.
    Parameters
    ----------
    ls
        Array with angular momenta of the basis functions.
    Returns
    -------
    permutation
        Reordering of the basis functions.
    """
    local_orders = {
        # Dictionary with reordering of the pure functions to match HORTON 2
        # conventions.
        0: np.array([0]),
        1: np.array([1, 2, 0]),
        2: np.array([2, 3, 1, 4, 0]),
        3: np.array([3, 4, 2, 5, 1, 6, 0]),
        4: np.array([4, 5, 3, 6, 2, 7, 1, 8, 0]),
    }
    result = []
    for l in ls:
        result.extend(local_orders[l] + len(result))
    return np.array(result)


def dump_spline(data, key, y, setup, l):
    """Convert a spline from a GPAW atom setup.
    Parameters
    ----------
    data
        Dictionary in which the spline is stored.
    key
        Used for making dictionary keys.
    y
        Function values at the spline grid points.
    setup
        The GPAW setup to which this spline belongs.
    l
        Angular momentum.
    """
    # Radial grid parameters
    a = setup.rgd.a
    b = setup.rgd.b
    rcut = max(setup.rcut_j)
    # The following is the size of the grid within the muffin tin sphere.
    size_short = int(np.ceil(rcut / (a + b * rcut)))

    # Create radial grid.
    rtf = HyperbolicRTransform(a, b)
    odg = OneDGrid(np.arange(size_short), np.ones(size_short), (0, size_short))
    rad_short = rtf.transform_1d_grid(odg)
    # Sanity checks
    assert_allclose(rad_short.points, setup.rgd.r_g[:size_short])
    assert_allclose(rad_short.weights, setup.rgd.dr_g[:size_short])

    # Correct normalization and create spline.
    ycorrected = y * np.sqrt((2 * l + 1) / np.pi) / 2
    cs_short = CubicSpline(rad_short.points, ycorrected[:size_short], bc_type="natural")

    # Radial grid within the muffin tin sphere
    data[key + ("radgrid",)] = rad_short
    # Cubic spline
    data[key + ("spline",)] = cs_short
    # Radius of the sphere.
    data[key + ("rcut",)] = rcut


def compute_augmentation_spheres(uniform_data, setups, atoms, atnums, atcoords):
    """Compute the density density corrections within the muffin tin spheres on grids.
    Parametes
    ---------
    uniform_data, setups, atoms
        Data generated by get_uniform_grid_data and get_atomic_grid_data.
    atnums
        Atomic numbers
    atcoords
        Atomic (nuclear) coordinates.
    All results are stored in the atoms argument.
    """
    w = abs(np.linalg.det(uniform_data["grid_vecs"]))
    nelec_pseudo = uniform_data["pseudo_density"].sum() * w
    if uniform_data["nspins"] == 2:
        spin_pseudo = uniform_data["pseudo_spindensity"].sum() * w
    natom = len(atnums)

    # Charge corrections are also computed here as a double check.
    qcors = uniform_data["charge_corrections"]
    myqcors = np.zeros(natom)
    if uniform_data["nspins"] == 2:
        sqcors = uniform_data["spincharge_corrections"]
        mysqcors = np.zeros(natom)
    else:
        sqcors = None
        mysqcors = None

    print("  ~~~~~~~  ~~~~~~~~~~~~~  ~~~~~~~~~~~~~  ~~~~~~~~~~~~~")
    print("     Atom  DensPart QCor      GPAW QCor          Error")
    print("  ~~~~~~~  ~~~~~~~~~~~~~  ~~~~~~~~~~~~~  ~~~~~~~~~~~~~")

    for iatom, atom_data in enumerate(atoms):
        setup_data = setups[atom_data["id_setup"]]

        # Do the actual nasty work...
        atgrid_short = eval_correction(atom_data, setup_data)
        atom_data["grid_points"] = atgrid_short.points + atcoords[iatom]
        atom_data["grid_weights"] = atgrid_short.weights

        # Add things up and compare.
        # - core part
        myqcors[iatom] = (
            atgrid_short.integrate(atom_data["density_c_cor"]) - atnums[iatom]
        )
        # - valence part
        vcor = atgrid_short.integrate(atom_data["density_v_cor"])
        myqcors[iatom] += vcor
        print(
            "  {:2d} {:4d}   {:12.7f}   {:12.7f}   {:12.5e}".format(
                atnums[iatom],
                iatom,
                myqcors[iatom],
                qcors[iatom],
                myqcors[iatom] - qcors[iatom],
            )
        )

        if sqcors is not None:
            mysqcors[iatom] = atgrid_short.integrate(atom_data["spindensity_v_cor"])
            print(
                "spin      {:12.7f}   {:12.7f}   {:12.5e}".format(
                    mysqcors[iatom], sqcors[iatom], mysqcors[iatom] - sqcors[iatom]
                )
            )

    print("  ~~~~~~~  ~~~~~~~~~~~~~  ~~~~~~~~~~~~~  ~~~~~~~~~~~~~")

    # Checks on the total charge
    print("  GPAW total charge:     {:10.3e}".format(nelec_pseudo + qcors.sum()))
    print("  DensPart total charge: {:10.3e}".format(nelec_pseudo + myqcors.sum()))
    assert_allclose(qcors, myqcors)
    if sqcors is not None:
        print("  GPAW total spin:       {:10.3e}".format(spin_pseudo + sqcors.sum()))
        print("  DensPart total spin:   {:10.3e}".format(spin_pseudo + mysqcors.sum()))
        assert_allclose(sqcors, mysqcors)


def eval_correction(atom_data, setup_data):
    """Compute the pseudo to all-electron corrections for one muffin-tin sphere.
    Parameters
    ----------
    atom_data
        Dictionary with the density matrices.
    setup_data
        Atomic (basis) functions stored on radial grids.
    Returns
    -------
    grid
        The atomic grid for integrations in the muffin tin sphere.
    Notes
    -----
    Conventions used in variable names, following GPAW conventions:
    - with t = pseudo
    - without t = all-electron
    - c = core
    - v = valence
    """
    # Setup atomic grid within the muffin tin sphere.
    radgrid = setup_data[("nc", "radgrid")]
    ls = setup_data["ls"]
    lmax = max(ls)
    # Twice lmax is used for the degree of the angular grid, because we include products
    # of two orbitals up to angular momentum lmax. Those products have up to angular
    # momentum 2 * lmax.
    grid = AtomGrid(
        radgrid,
        degrees=[2 * lmax] * radgrid.size,
    )

    d = np.linalg.norm(grid.points, axis=1)

    # Compute the core density correction.
    cs_nc = setup_data[("nc", "spline")]
    cs_nct = setup_data[("nct", "spline")]
    atom_data["density_c"] = cs_nc(d)
    atom_data["density_ct"] = cs_nct(d)
    atom_data["density_c_cor"] = atom_data["density_c"] - atom_data["density_ct"]

    # Compute real spherical harmonics (with Racah normalization) on the grid.
    polys = np.zeros(((lmax + 1) ** 2 - 1, grid.size), float)
    polys[0] = grid.points[:, 2]
    polys[1] = grid.points[:, 0]
    polys[2] = grid.points[:, 1]
    spherical_harmonics(polys, lmax, racah=True)

    # Evaluate each pseudo and ae basis function in the atomic grid.
    basis_fns = []
    basist_fns = []
    for iradial, l in enumerate(ls):
        # Evaluate radial functions.
        phi = setup_data[("phi", iradial, "spline")]
        basis = phi(d)
        phit = setup_data[("phit", iradial, "spline")]
        basist = phit(d)

        # Multiply with the corresponding spherical harmonics
        if l == 0:
            basis_fns.append(basis)
            basist_fns.append(basist)
        else:
            # Number of spherical harmonics and offset in the polys array.
            nfn = 2 * l + 1
            offset = l ** 2 - 1
            for ifn in range(nfn):
                poly = polys[offset + ifn]
                basis_fns.append(basis * poly)
                basist_fns.append(basist * poly)

    # Sanity check:
    # Construct the local overlap matrix and compare to the one taken from GPAW.
    olp = np.zeros((len(basis_fns), len(basis_fns)))
    olpt = np.zeros((len(basis_fns), len(basis_fns)))
    for ibasis0, (phi0, phit0) in enumerate(zip(basis_fns, basist_fns)):
        for ibasis1 in range(ibasis0 + 1):
            phi1 = basis_fns[ibasis1]
            phit1 = basist_fns[ibasis1]
            olp[ibasis0, ibasis1] = grid.integrate(phi0 * phi1)
            olp[ibasis1, ibasis0] = olp[ibasis0, ibasis1]
            olpt[ibasis0, ibasis1] = grid.integrate(phit0 * phit1)
            olpt[ibasis1, ibasis0] = olpt[ibasis0, ibasis1]
    assert_allclose(olp - olpt, setup_data["overlap"], atol=1e-10)

    # Load the atomic density matrix
    dm = atom_data["dm"]
    if "spindm" in atom_data:
        spindm = atom_data["spindm"]
    else:
        spindm = None

    # Loop over all pairs of basis functions and add product times density matrix coeff
    density_v = np.zeros(grid.size)
    density_vt = np.zeros(grid.size)
    if spindm is not None:
        spindensity_v = np.zeros(grid.size)
        spindensity_vt = np.zeros(grid.size)
    for ibasis0, (phi0, phit0) in enumerate(zip(basis_fns, basist_fns)):
        for ibasis1 in range(ibasis0 + 1):
            phi1 = basis_fns[ibasis1]
            phit1 = basist_fns[ibasis1]
            factor = (ibasis0 != ibasis1) + 1
            density_v += factor * dm[ibasis0, ibasis1] * phi0 * phi1
            density_vt += factor * dm[ibasis0, ibasis1] * phit0 * phit1
            if spindm is not None:
                spindensity_v += factor * spindm[ibasis0, ibasis1] * phi0 * phi1
                spindensity_vt += factor * spindm[ibasis0, ibasis1] * phit0 * phit1

    # Store electronic valence densities
    density_v_cor = density_v - density_vt
    # Sanity check
    assert np.allclose(
        grid.integrate(density_v_cor), np.dot((olp - olpt).ravel(), dm.ravel())
    )
    atom_data["density_v"] = density_v
    atom_data["density_vt"] = density_vt
    atom_data["density_v_cor"] = density_v_cor
    if spindm is not None:
        spindensity_v_cor = spindensity_v - spindensity_vt
        # Sanity check
        assert np.allclose(
            grid.integrate(spindensity_v_cor),
            np.dot((olp - olpt).ravel(), spindm.ravel()),
        )
        atom_data["spindensity_v"] = spindensity_v
        atom_data["spindensity_vt"] = spindensity_vt
        atom_data["spindensity_v_cor"] = spindensity_v_cor

    return grid


def compute_uniform_points(uniform_data):
    """Compute the trivial positions and weights of the uniform grid points."""
    # construct array with point coordinates
    shape = uniform_data["shape"]
    grid_rvecs = uniform_data["grid_vecs"]
    points = np.zeros(tuple(shape) + (3,))
    # pylint: disable=too-many-function-args
    points += np.outer(np.arange(shape[0]), grid_rvecs[0]).reshape(shape[0], 1, 1, 3)
    points += np.outer(np.arange(shape[1]), grid_rvecs[1]).reshape(1, shape[1], 1, 3)
    points += np.outer(np.arange(shape[2]), grid_rvecs[2]).reshape(1, 1, shape[2], 3)

    # Check some points.
    npoint = points.size // 3
    for ipoint in range(0, npoint, npoint // 100):
        # pylint: disable=unbalanced-tuple-unpacking
        i0, i1, i2 = np.unravel_index(ipoint, shape)
        assert np.allclose(
            points[i0, i1, i2],
            i0 * grid_rvecs[0] + i1 * grid_rvecs[1] + i2 * grid_rvecs[2],
        )
    points.shape = (-1, 3)
    weights = np.empty(len(points))
    weights.fill(abs(np.linalg.det(grid_rvecs)))

    uniform_data["grid_points"] = points
    uniform_data["grid_weights"] = weights


def denspart_conventions(uniform_data, atoms):
    """Convert all result from all the above functions into a format suitable for denspart.
    Parameters
    ----------
    uniform_data
        Dictionary with detailed data from the uniform grid of a GPAW calculation.
    atoms
        List with dicationaries with atomic grid data.
    Returns
    -------
    density
        Dictionary with just the data needed for running denspart.
    """
    grid_parts = [GridPart(uniform_data, "pseudo_density")]
    print("  Uniform grid size:", grid_parts[0].density.size)
    for atom in atoms:
        grid_parts.append(GridPart(atom, "density_c_cor", "density_v_cor"))
        print("  Atom grid size:", grid_parts[-1].density.size)
    result = {
        "points": np.concatenate([gp.points for gp in grid_parts]),
        "weights": np.concatenate([gp.weights for gp in grid_parts]),
        "density": np.concatenate([gp.density for gp in grid_parts]),
    }
    print("  Total grid size:", result["density"].size)

    if uniform_data["nspins"] == 2:
        spin_grid_parts = [GridPart(uniform_data, "pseudo_spindensity")]
        for atom in atoms:
            spin_grid_parts.append(GridPart(atom, "spindensity_v_cor"))
        result["spindensity"] = np.concatenate([gp.density for gp in grid_parts])

    return result


class GridPart:
    """Helper class for collecting density grid data."""

    def __init__(self, data, *densnames):
        self.points = data["grid_points"].reshape(-1, 3)
        self.weights = data["grid_weights"].ravel()
        self.density = sum(data[name] for name in densnames).ravel()




def calc_f0j(cell_mat_m, element_symbols, positions, index_vec_h, symm_mats_vecs, computation_dict=None, restart=None, save='gpaw.gpw', explicit_core=True):
    """
    Calculate the aspherical atomic form factors from a density grid in the python package gpaw
    for each reciprocal lattice vector present in index_vec_h.
    """
    if computation_dict is None:
        computation_dict = {'xc': 'PBE', 'txt': 'gpaw.txt', 'h': 0.15, 'setups': 'paw'}
    else:
        computation_dict = computation_dict.copy()
    if 'gridinterpolation' in computation_dict:
        gridinterpolation = computation_dict['interpolation']
        #print(f'interpolation set to {interpolation}')
        del(computation_dict['interpolation'])
    else:
        interpolation = 2
    if 'average_symmequiv' in computation_dict:
        average_symmequiv = computation_dict['average_symmequiv']
        #print(f'average symmetry equivalents: {average_symmequiv}')
        del(computation_dict['average_symmequiv'])
    else:
        average_symmequiv = False
    if 'skip_symm' in computation_dict:
        #assert len(computation_dict['skip_symm']) == 0 or average_symmequiv, 'skip_symm does need average_symmequiv' 
        skip_symm = computation_dict['skip_symm']
        del(computation_dict['skip_symm'])
    else:
        skip_symm = {}
    if 'magmoms' in computation_dict:
        magmoms = computation_dict['magmoms']
        del(computation_dict['magmoms'])
    else:
        magmoms = None
    if 'denspart_gtol' in computation_dict:
        denspart_gtol = computation_dict['denspart_gtol']
        del(computation_dict['denspart_gtol'])
    else:
        denspart_gtol = 1e-8
    if 'denspart_maxiter' in computation_dict:
        denspart_maxiter = computation_dict['denspart_maxiter']
        del(computation_dict['denspart_maxiter'])
    else:
        denspart_maxiter = 1000
    if 'denspart_density_cutoff' in computation_dict:
        denspart_density_cutoff = computation_dict['denspart_density_cutoff']
        del(computation_dict['denspart_density_cutoff'])
    else:
        denspart_density_cutoff = 1e-10

    #assert not (not average_symmequiv and not do_not_move)
    symm_positions, symm_symbols, f0j_indexes, magmoms_symm = expand_symm_unique(element_symbols,
                                                                                 np.array(positions),
                                                                                 np.array(cell_mat_m),
                                                                                 (np.array(symm_mats_vecs[0]), np.array(symm_mats_vecs[1])),
                                                                                 skip_symm=skip_symm,
                                                                                 magmoms=magmoms)
    e_change = True
    if restart is None:
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
                atoms, calc = gpaw.restart(restart, txt=computation_dict['txt'], xc=computation_dict['xc'])
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

    density = np.array(calc.get_all_electron_density(gridrefinement=gridinterpolation, skip_core=explicit_core))
    #if explicit_core:
    #    n_elec = sum([setup.Z for setup in calc.setups]) - sum(setup.Nc for setup in calc.density.setups)
    #else:
    n_elec = sum([setup.Z for setup in calc.setups])
    density *= n_elec / density.sum()
    if save is not None and e_change:
        try:
            calc.write(save, mode='all')
        except:
            print('  Could not save to file')

    print('  calculated density with energy', e1)

    with open('denspart.txt', 'w') as fo:
        with contextlib.redirect_stdout(fo):
            dp_data = prepare_input(atoms, calc)    

    dp_grid = PeriodicGrid(
        dp_data['points'], dp_data['weights'], dp_data['cellvecs'], wrap=True
    )
    print('  using denspart to fit the MBIS partitioning')
    #with open('denspart.txt', 'a') as fo:
        #with contextlib.redirect_stdout(fo):
    print("MBIS partitioning --")
    if os.path.exists('denspart.npz') and restart is not None and False:
        with open('denspart.npz', 'rb') as fo:
            dp_dict = dict(np.load(fo))
        dp_dict['atcoords'] = atoms.positions
        pro_model_init = MBISProModel.from_dict(dp_dict)
    else:
        pro_model_init = MBISProModel.from_geometry(dp_data['atnums'], dp_data['atcoords'])


    pro_model, localgrids = optimize_reduce_pro_model(
        pro_model_init,
        dp_grid,
        dp_data['density'],
        denspart_gtol,
        denspart_maxiter,
        denspart_density_cutoff,
    )

    with open('denspart.npz', 'wb') as fo:
        np.savez(fo, **pro_model.to_dict())
    
    if gridinterpolation == 1:
        coords = calc.density.gd.get_grid_point_coordinates()
        coords = np.rollaxis(coords, 0, 4).reshape(np.prod(coords.shape[1:]), 3)
    elif gridinterpolation == 2:
        coords = calc.density.finegd.get_grid_point_coordinates()
        coords = np.rollaxis(coords, 0, 4).reshape(np.prod(coords.shape[1:]), 3)
    elif gridinterpolation == 4:
        coords = calc.density.finegd.get_grid_point_coordinates()

        delta_x = coords[0, 1, 0, 0] - coords[0, 0, 0, 0] 
        delta_y = coords[1, 0, 1, 0] - coords[0, 0, 0, 0] 
        delta_z = coords[2, 0, 0, 1] - coords[0, 0, 0, 0] 
        coords = np.repeat(coords, 2, axis=1)
        coords = np.repeat(coords, 2, axis=2)
        coords = np.repeat(coords, 2, axis=3)
        coords[0, 1::2, :, :] += delta_x / 2
        coords[1, :, 1::2, :] += delta_y / 2
        coords[2, :, :, 1::2] += delta_z / 2
        coords = np.rollaxis(coords, 0, 4).reshape(np.prod(coords.shape[1:]), 3)
    else:
        raise NotImplementedError('only possible values for gridinterpolation are 1, 2 and 4')
    
    assert -density.shape[0] // 2 < index_vec_h[:,0].min(), 'Your gridspacing is too large.'
    assert density.shape[0] // 2 > index_vec_h[:,0].max(), 'Your gridspacing is too large.'
    assert -density.shape[1] // 2 < index_vec_h[:,1].min(), 'Your gridspacing is too large.'
    assert density.shape[1] // 2 > index_vec_h[:,1].max(), 'Your gridspacing is too large.'
    assert -density.shape[2] // 2 < index_vec_h[:,2].min(), 'Your gridspacing is too large.'
    assert density.shape[2] // 2 > index_vec_h[:,2].max(), 'Your gridspacing is too large.'
    overall_prodensity = np.zeros_like(density)
    for xi, yi, zi, in product(range(-1, 2, 1), repeat=3):
        offset = xi * dp_grid.realvecs[0] + yi * dp_grid.realvecs[1] + zi * dp_grid.realvecs[2]
        overall_prodensity += sum(pro_model.compute_proatom(i, coords + offset).reshape(density.shape) for i in range(len(atoms)))

    f0j = np.zeros((symm_mats_vecs[0].shape[0], positions.shape[0], index_vec_h.shape[0]), dtype=np.complex128)
    print('  splitting up atoms with MBIS')
    if average_symmequiv:
        h, k, l = np.meshgrid(*map(lambda n: np.fft.fftfreq(n, 1/n).astype(np.int64), density.shape), indexing='ij')
        for atom_index, symm_atom_indexes in enumerate(f0j_indexes):
            f0j_sum = np.zeros_like(h, dtype=np.complex128)
            for symm_matrix, symm_atom_index in zip(symm_mats_vecs[0], symm_atom_indexes):
                pro_dens = np.zeros_like(density)
                for xi, yi, zi, in product(range(-1, 2, 1), repeat=3):
                    offset = xi * dp_grid.realvecs[0] + yi * dp_grid.realvecs[1] + zi * dp_grid.realvecs[2]
                    pro_dens += pro_model.compute_proatom(symm_atom_index, coords + offset).reshape(density.shape)
                atom_density = density * pro_dens / overall_prodensity
                frac_position = symm_positions[symm_atom_index]
                h_rot, k_rot, l_rot = np.einsum('xy, y... -> x...', symm_matrix, np.array((h, k, l))).astype(np.int64)
                phase_to_zero = np.exp(-2j * np.pi * (frac_position[0] * h + frac_position[1] * k + frac_position[2] * l))
                f0j_sum += (np.fft.ifftn(atom_density) * phase_to_zero * np.prod(h.shape))[h_rot, k_rot, l_rot]
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
                    pro_dens = np.zeros_like(density)
                    for xi, yi, zi, in product(range(-1, 2, 1), repeat=3):
                        offset = xi * dp_grid.realvecs[0] + yi * dp_grid.realvecs[1] + zi * dp_grid.realvecs[2]
                        pro_dens += pro_model.compute_proatom(symm_atom_index, coords + offset).reshape(density.shape)
                    atom_density = density * pro_dens / overall_prodensity
                    frac_position = symm_positions[symm_atom_index]
                    phase_to_zero = np.exp(-2j * np.pi * (frac_position[0] * h_vec + frac_position[1] * k_vec + frac_position[2] * l_vec))
                    f0j[symm_index, atom_index, :] = ((np.fft.ifftn(atom_density) * np.prod(density.shape))[h_vec, k_vec, l_vec] * phase_to_zero).copy()
                    already_known[symm_atom_index] = (symm_index, atom_index)
    del(calc)
    del(atoms)
    return f0j



def calculate_f0j_core(cell_mat_m, element_symbols, positions, index_vec_h, symm_mats_vecs):
    raise NotImplementedError('Core partitioning is not implemented in MBIS')