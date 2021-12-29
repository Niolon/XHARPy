from scipy.optimize import minimize
import numpy as np
import jax
import jax.numpy as jnp
from .conversion import calc_sin_theta_ov_lambda

def calc_weights(parameters, c_mode, intensity_pos, i_calc, esd_int, sintheta_ov_lambda=None):
    a, b, c, d, e, f = parameters

    if c_mode > 0:
        q = jnp.exp(c * (sintheta_ov_lambda)**2)
    elif c_mode < 0:
        q = 1 - jnp.exp(c * (sintheta_ov_lambda)**2)
    else:
        q = 1
    p = f * intensity_pos + (1 - f) * i_calc

    return q / (esd_int**2 + (a * p)**2 + b*p + d + e * sintheta_ov_lambda)

# TODO: Implement weightig refinement
