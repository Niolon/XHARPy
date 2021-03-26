
import jax.numpy as jnp


def ucif2ucart(cell_mat_m, u_mats):
    # see R. W. Grosse-Kunstleve and P. D. Adams J. Appl. Cryst 2002, p.478 eq. 3a + 4a
    cell_mat_f = jnp.linalg.inv(cell_mat_m).T
    cell_mat_n = jnp.eye(3) * jnp.linalg.norm(cell_mat_f, axis=1)

    u_star = jnp.einsum('ab, zbc, cd -> zad', cell_mat_n, u_mats, cell_mat_n.T)
    return jnp.einsum('ab, zbc, cd -> zad', cell_mat_m, u_star, cell_mat_m.T)


def cell_constants_to_M(a, b, c, alpha, beta, gamma):
    """
    Generates a matrix with the three lattice vectors as lines
    unit of length will be as given by the cell constants
    """
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


def calc_sin_theta_ov_lambda(cell_mat_f, index_vec_h):
    return jnp.linalg.norm(jnp.einsum('xy, zy -> zx', cell_mat_f, index_vec_h), axis=1) / 2 
