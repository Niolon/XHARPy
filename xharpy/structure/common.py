import warnings
from collections import namedtuple
from typing import Tuple, Union
try:
    import jax.numpy as jnp
except:
    warnings.warn('Jax was not found. Refinement will not be available')

from dataclasses import dataclass

AtomInstructions = namedtuple('AtomInstructions', [
    'name', # Name of atom, supposed to be unique
    'element', # Element symbol, e.g. 'Ca'
    'dispersion_real', # dispersion correction f'
    'dispersion_imag', # dispersion correction f''
    'xyz', # fractional coordinates
    'uij', # uij parameters U11, U22, U33, U23, U13, U12
    'cijk', # Gram Charlier 3rd Order C111, C222, C333, C112, C122, C113, C133,
            # C223, C233, C123,
    'dijkl', # Gram Charlier 4th Order D1111, D2222, D3333, D1112, D1222, D1113,
             # D1333, D2223, D2333, D1122, D1133, D2233, D1123, D1223, D1233
    'occupancy' # the occupancy
], defaults= [None, None, None, None, None, None, None, None])

@dataclass(frozen=True)
class RefinedParameter:
    par_index: int
    multiplicator: float = 1.0
    added_value: float = 0.0
    special_position: bool = False

    def resolve(self, parameters):
        return self.multiplicator * parameters[self.par_index] + self.added_value
    
    def resolve_esd(self, var_cov_mat):
        return jnp.abs(self.multiplicator) * jnp.sqrt(var_cov_mat[self.par_index, self.par_index])

@dataclass(frozen=True)
class FixedParameter:
    value: float
    special_position: bool = False

    def resolve(self, parameters):
        return self.value

    def resolve_esd(self, var_cov_mat):
        return jnp.nan # One could pick zero, but this should indicate that an error is not defined

@dataclass(frozen=True)
class MultiIndexParameter:
    par_indexes: Tuple[int]
    multiplicators: Tuple[float]
    added_value: float

    def resolve(self, parameters):
        multiplicators = jnp.array(self.multiplicators)
        par_values = jnp.take(parameters, jnp.array(self.par_indexes), axis=0)
        return jnp.sum(multiplicators * par_values) + self.added_value
    
    def resolve_esd(self, var_cov_mat):
        jac = jnp.array(self.multiplicators)
        indexes = jnp.array(self.par_indexes)
        return jnp.sqrt(jnp.sqrt(jac[None, :] @ var_cov_mat[indexes][: , indexes] @ jac[None, :].T))[0,0]

Parameter = Union[RefinedParameter, FixedParameter, MultiIndexParameter]

@dataclass(frozen=True)
class Array:
    parameter_tuple: Tuple[Parameter]
    derived: bool = False
    def resolve(self, parameters):
        return jnp.array([par.resolve(parameters) for par in self.parameter_tuple])

    def resolve_esd(self, var_cov_mat):
        return jnp.array([par.resolve_esd(var_cov_mat) for par in self.parameter_tuple])
