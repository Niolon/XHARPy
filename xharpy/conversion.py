
import jax.numpy as jnp


def ucif2ucart(cell_mat_m: jnp.ndarray, u_mats: jnp.ndarray) -> jnp.ndarray:
    """Calculate anistropic displacement matrices in the cartesian convention
    from the displacement matrices in the cif convention
    see: R. W. Grosse-Kunstleve and P. D. Adams J. Appl. Cryst 2002, p.478
    eq. 3a + 4a

    Parameters
    ----------
    cell_mat_m : jnp.ndarray
        (3,3) array containing the cell vectors as row vectors
    u_mats : jnp.ndarray
        (N, 3, 3) array containing the anisotropic displacement matrices in
        cif format

    Returns
    -------
    u_cart: jnp.ndarray
        (N, 3, 3) array of the matrices in cartesian convention
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
        (3, 3) array containing the cell vectors as row vectors
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
    """Calculate the resolution in sin(theta)/lambda for the given set of miller
    indicees

    Parameters
    ----------
    cell_mat_f : jnp.ndarray
        (3, 3) array containing the reciprocal lattice vectors
    index_vec_h : jnp.ndarray
        (H, 3) array of Miller indicees of reflections


    Returns
    -------
    sin_theta_ov_lambda: jnp.ndarray
        (H) array containing the calculated values
    """
    return jnp.linalg.norm(jnp.einsum('xy, zy -> zx', cell_mat_f, index_vec_h), axis=1) / 2 
