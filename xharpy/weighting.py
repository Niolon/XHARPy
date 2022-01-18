"""Is meant as the module for handling weights in XHARPy. Currently 
only contains a function to calculate weights from given parameters."""

from scipy.optimize import minimize
import numpy as np
import jax
import jax.numpy as jnp
from .conversion import calc_sin_theta_ov_lambda

def calc_weights(
    wght_parameters: jnp.ndarray,
    c_mode: float,
    intensity_pos: jnp.ndarray,
    i_calc: jnp.ndarray,
    esd_int: jnp.ndarray,
    sintheta_ov_lambda: jnp.ndarray=None
) -> jnp.ndarray:
    """Calculate weights for given arguments

    Parameters
    ----------
    wght_parameters : jnp.ndarray
        size (6) array containing the weight parameters
    c_mode : float
        in order to be able to call this with derivatives c_mode needs
        to be called separately
    intensity_pos : jnp.ndarray
        experimental observed intensities
    i_calc : jnp.ndarray
        calculated intensities from the refinement
    esd_int : jnp.ndarray
        estimated standard deviations of the observed intensities
    sintheta_ov_lambda : jnp.ndarray, optional
        if c_mode != 0 needed sin(theta)/lambda values, by default None

    Returns
    -------
    jnp.ndarray
        weights for the individual reflections
    """
    a, b, c, d, e, f = wght_parameters

    if c_mode > 0:
        q = jnp.exp(c * (sintheta_ov_lambda)**2)
    elif c_mode < 0:
        q = 1 - jnp.exp(c * (sintheta_ov_lambda)**2)
    else:
        q = 1
    p = f * intensity_pos + (1 - f) * i_calc

    return q / (esd_int**2 + (a * p)**2 + b*p + d + e * sintheta_ov_lambda)

