from scipy.optimize import minimize
import numpy as np
import jax
import jax.numpy as jnp
from conversion import calc_sin_theta_ov_lambda

def calc_weights(parameters, c_mode, intensity_pos, i_calc, stderr, sintheta_ov_lambda=None):
    a, b, c, d, e, f = parameters

    if c_mode > 0:
        q = jnp.exp(c * (sintheta_ov_lambda)**2)
    elif c_mode < 0:
        q = 1 - jnp.exp(c * (sintheta_ov_lambda)**2)
    else:
        q = 1
    p = f * intensity_pos + (1 - f) * i_calc

    return q / (stderr**2 + (a * p)**2 + b*p + d + e *sintheta_ov_lambda)


def weight_difference(refined_parameters, non_refined_parameters, intensity, i_calc, stderr, sintheta_ov_lambda, c_mode, n_p):
    n_i = len(intensity)
    parameters = jnp.zeros(6)
    ref_indexes = jnp.arange(len(refined_parameters))
    nonref_indexes = jnp.arange(-len(non_refined_parameters), 0)
    parameters = jax.ops.index_update(parameters, ref_indexes, refined_parameters)
    parameters = jax.ops.index_update(parameters, nonref_indexes, non_refined_parameters)

    weights = calc_weights(parameters, c_mode, intensity, i_calc, stderr, sintheta_ov_lambda)
    
    gof = jnp.sum((weights * (intensity - i_calc)**2)**2) / (n_i - n_p)

    return (gof - 1)**2
    

def refine_weighting(refined_parameters, non_refined_parameters, intensity, i_calc, stderr, sintheta_ov_lambda, c_mode, n_p):
    intensity_pos = intensity.copy()
    intensity_pos[intensity_pos < 0] = 0
    weight_fun_jit = jax.jit(weight_difference, static_argnums=(1, 2, 3, 4, 5, 6 ,7))
    x = minimize(weight_fun_jit, x0=refined_parameters, args=(non_refined_parameters, intensity_pos, i_calc, stderr, sintheta_ov_lambda, c_mode, n_p))
    print(f'calculated value for sqrt((gof - 1)**2) as: {np.sqrt(x.fun):7.6f}')
    return x.x

