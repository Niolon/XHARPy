"""This module contains conversions, that are needed in different parts of the 
library. It is not meant to import from anywhere within XHARPy so that its
functions can be import without circular imports"""

from .common_jax import jnp
from typing import List, Tuple, Dict, Optional
import numpy as np


def ucif2ucart(cell_mat_m: jnp.ndarray, u_mats: jnp.ndarray) -> jnp.ndarray:
    """Calculate anisotropic displacement matrices in the cartesian convention
    from the displacement matrices in the cif convention
    see: R. W. Grosse-Kunstleve and P. D. Adams J. Appl. Cryst 2002, p.478
    eq. 3a + 4a

    Parameters
    ----------
    cell_mat_m : jnp.ndarray
        size (3,3) array containing the cell vectors as row vectors
    u_mats : jnp.ndarray
        size (N, 3, 3) array containing the anisotropic displacement matrices in
        cif format

    Returns
    -------
    u_cart: jnp.ndarray
        size (N, 3, 3) array of the matrices in cartesian convention
    """
    # 
    cell_mat_f = jnp.linalg.inv(cell_mat_m).T
    cell_mat_n = jnp.eye(3) * jnp.linalg.norm(cell_mat_f, axis=1)

    u_star = jnp.einsum('ab, zbc, cd -> zad', cell_mat_n, u_mats, cell_mat_n.T)
    return jnp.einsum('ab, zbc, cd -> zad', cell_mat_m, u_star, cell_mat_m.T)


def cell_constants_to_M(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    crystal_system: str = 'triclinic'
):
    """Generates a matrix with the three lattice vectors as row vectors

    Parameters
    ----------
    a : float
        cell constant a in Angstroem
    b : float
        cell constant b in Angstroem
    c : float
        cell constant c in Angstroem
    alpha : float
        cell angle alpha in degree
    beta : float
        cell angle beta in degree
    gamma : float
        cell angle gamma in degree
    crystal_system : str, optional
        Crystal system of the evaluated structure. Possible values are: 
        'triclinic', 'monoclinic' 'orthorhombic', 'tetragonal', 'hexagonal',
        'trigonal' and 'cubic'. Does not make a difference for the calculation
        of the matrix, but does make a difference for the derivatives to the
        cell parameters

    Returns
    -------
    cell_mat_m: jnp.ndarray
        size (3, 3) array containing the cell vectors as row vectors
    """
    if crystal_system == 'monoclinic':
        alpha = 90.0
        gamma = 90.0
    elif crystal_system == 'orthorhombic':
        alpha = 90.0
        beta = 90.0
        gamma = 90.0
    elif crystal_system == 'tetragonal':
        b = a
        alpha = 90.0
        beta = 90.0
        gamma = 90.0
    elif crystal_system in ('hexagonal', 'trigonal'):
        b = a
        alpha = 90.0
        beta = 90.0
        gamma = 120.0
    elif crystal_system == 'cubic':
        b = a
        c = a
        alpha = 90.0
        beta = 90.0
        gamma = 90.0
    alpha = alpha / 180.0 * jnp.pi
    beta = beta / 180.0 * jnp.pi
    gamma = gamma / 180.0 * jnp.pi
    M = jnp.array(
        [
            [
                a,
                0,
                0
            ],
            [
                b * jnp.cos(gamma),
                b * jnp.sin(gamma),
                0
            ],
            [
                c * jnp.cos(beta),
                c * (jnp.cos(alpha) - jnp.cos(gamma) * jnp.cos(beta)) / jnp.sin(gamma),
                c / jnp.sin(gamma) * jnp.sqrt(1.0 - jnp.cos(alpha)**2 - jnp.cos(beta)**2
                                            - jnp.cos(gamma)**2
                                            + 2 * jnp.cos(alpha) * jnp.cos(beta) * jnp.cos(gamma))
            ]
        ]
    )
    return M.T


def calc_sin_theta_ov_lambda(
    cell_mat_f: jnp.ndarray,
    index_vec_h: jnp.ndarray
) -> jnp.ndarray:
    """Calculate the resolution in sin(theta)/lambda for the given set of Miller
    indicees

    Parameters
    ----------
    cell_mat_f : jnp.ndarray
        size (3, 3) array containing the reciprocal lattice vectors
    index_vec_h : jnp.ndarray
        size (H, 3) array of Miller indicees of reflections


    Returns
    -------
    sin_theta_ov_lambda: jnp.ndarray
        size (H) array containing the calculated resolution values
    """
    return jnp.linalg.norm(jnp.einsum('xy, zy -> zx', cell_mat_f, index_vec_h), axis=1) / 2 


def expand_symm_unique(
        type_symbols: List[str],
        coordinates: np.ndarray,
        cell_mat_m: np.ndarray,
        symm_mats_vec: Tuple[np.ndarray, np.ndarray],
        skip_symm: Dict[str, List[int]] = {},
        magmoms: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[str],
               np.ndarray, Optional[np.ndarray]]:
    """Expand the type_symbols and coordinates for one complete unit cell.
    Atoms on special positions appear only once. For disorder on a special
    position use skip_symm.


    Parameters
    ----------
    type_symbols : List[str]
        Element symbols of the atoms in the asymmetric unit
    coordinates : npt.NDArray[np.float64]
        size (N, 3) array of fractional atomic coordinates
    cell_mat_m : npt.NDArray[np.float64]
        Matrix with cell vectors as column vectors, (Angstroem)
    symm_mats_vec : Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        size (K, 3, 3) array of symmetry matrices and size (K, 3) array
        of translation vectors for all symmetry elements in the unit cell
    skip_symm : Dict[str, List[int]], optional
        Symmetry elements with indexes given the list(s) in the dictionary
        values with not be applied to the respective atoms with the atom names
        given in the key(s). Indexes need to be identical to the ones in 
        symm_mats_vec., by default {}
    magmoms : Optional[npt.NDArray[np.float64]], optional
        Magnetic Moments. The enforced symmetry might not be correcz, by default
        None

    Returns
    -------
    symm_positions: npt.NDArray[np.float64]
        size (M, 3) array of all unique atom positions within the unit cell
    symm_symbols: List[str]
        ist of length M with element symbols for the unique atom positions
        within the unit cell
    reverse_indexes: npt.NDArray[np.float64]
        size (K, N) array with indexes mapping the unique atom positions back to 
        the individual symmetry elements and atom positions in the asymmetric 
        unit
    symm_magmoms: Optional[npt.NDArray[np.float64]]]
        magnetic moments for symmetry generated atoms. Undertested!
    """
    symm_mats_r, symm_vecs_t = symm_mats_vec
    pos_frac0 = coordinates % 1
    un_positions = np.zeros((0, 3))
    n_atoms = 0
    type_symbols_symm = []
    inv_indexes = []
    if magmoms is not None:
        magmoms_symm = []
    else:
        magmoms_symm = None
    # Only check atom with itself
    for atom_index, (pos0, type_symbol) in enumerate(zip(pos_frac0, type_symbols)):
        if atom_index in skip_symm:
            use_indexes = [i for i in range(symm_mats_r.shape[0]) if i not in skip_symm[atom_index]]
        else:
            use_indexes = list(range(symm_mats_r.shape[0]))
        symm_positions = (np.einsum(
            'kxy, y -> kx',
             symm_mats_r[use_indexes, :, :], pos0) + symm_vecs_t[use_indexes, :]
        ) % 1
        _, unique_indexes, inv_indexes_at = np.unique(
            np.round(np.einsum('xy, zy -> zx', cell_mat_m, symm_positions), 3),
            axis=0,
            return_index=True,
            return_inverse=True
        )
        un_positions = np.concatenate((un_positions, symm_positions[unique_indexes]))
        type_symbols_symm += [type_symbol] * unique_indexes.shape[0]
        if magmoms is not None:
            magmoms_symm += [magmoms[atom_index]] * unique_indexes.shape[0]
        inv_indexes.append(inv_indexes_at + n_atoms)
        n_atoms += unique_indexes.shape[0]
    if magmoms_symm is not None:
        magmoms_symm = np.array(magmoms_symm)
    return un_positions.copy(), type_symbols_symm, np.array(inv_indexes, dtype=object), magmoms_symm


