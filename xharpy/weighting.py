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


def weight_difference(refined_parameters, non_refined_parameters, intensity, i_calc, esd_int, sintheta_ov_lambda, c_mode, n_p, conditions):
    n_i = len(intensity)
    parameters = jnp.zeros(6)
    ref_indexes = jnp.arange(len(refined_parameters))
    nonref_indexes = jnp.arange(-len(non_refined_parameters), 0)
    parameters = jax.ops.index_update(parameters, ref_indexes, refined_parameters)
    parameters = jax.ops.index_update(parameters, nonref_indexes, non_refined_parameters)

    weights = calc_weights(parameters, c_mode, intensity, i_calc, esd_int, sintheta_ov_lambda)
    
    return jnp.mean((jnp.array([jnp.sqrt(jnp.sum(weights[condition] * (intensity[condition] -  i_calc[condition])**2) 
                               / (jnp.sum(condition)- n_p)) for condition in conditions]) - 1)**2)

    

def refine_weighting(refined_parameters, non_refined_parameters, intensity, i_calc, esd_int, sintheta_ov_lambda, c_mode, n_p):
    intensity_pos = intensity.copy()
    intensity_pos[intensity_pos < 0] = 0
    f_abs = np.sqrt(i_calc)
    limits = [0] +  [np.quantile(f_abs, quant) for quant in np.arange(0.1, 1.0, 0.1)] + [np.inf]
    conditions = [jnp.logical_and(f_abs >= limit1, f_abs < limit2) for limit1, limit2 in zip(limits[:-1], limits[1:])]
    weight_fun_jit = jax.jit(weight_difference, static_argnums=(1, 2, 3, 4, 5, 6 ,7, 8))
    grad_weight_fun = jax.grad(weight_fun_jit)
    x = minimize(weight_fun_jit,
                 x0=refined_parameters,
                 args=(non_refined_parameters, intensity_pos, i_calc, esd_int, sintheta_ov_lambda, c_mode, n_p, conditions),
                 jac=grad_weight_fun)
    print(f'calculated value for var(gof) as: {x.fun:7.6f}')
    return x.x

