from .common import Value, Array

from ..common_jax import jnp, jax
from ..conversion import cell_constants_to_M, ucif2ucart
from ..better_abc import ABC, ABCMeta, abstract_attribute, abstractmethod
from collections import namedtuple
from dataclasses import dataclass


UEquivCalculated = namedtuple('UEquivCalculated', [
    'atom_index',   # index of atom to set the U_equiv equal to 
    'multiplicator' # factor to multiply u_equiv with
])

UIso = namedtuple('Uiso',[
    'uiso'          # Parameter for Uiso can either be a fixed parameter
                    # or a refined Parameter
], defaults=[0.1])

class TFactor(metaclass=ABCMeta):
    adp_type: str = abstract_attribute()
    derived: bool = abstract_attribute()

    @abstractmethod
    def as_uij(self, parameters, **kwargs):
        return None
    
    @abstractmethod
    def uij_esd(self, var_cov_mat):
        return None

    @abstractmethod
    def u_equiv(self, parameters, **kwargs):
        return None

    @abstractmethod
    def u_equiv_esd(self, var_cov_mat, **kwargs):
        return None



@dataclass(frozen=True)
class AnisoTFactor(TFactor):
    uij_values: Array
    adp_type = 'Uani'
    derived = False

    def as_uij(self, parameters, **kwargs):
        return self.uij_values.resolve(parameters)

    def uij_esd(self, var_cov_mat):
        return self.uij_values.resolve_esd(var_cov_mat)

    def u_equiv(self, parameters, cell_mat_m, **kwargs):
        uij = self.uij_values.resolve(parameters)
        u_mats = uij[None, jnp.array([[0, 5, 4],
                                      [5, 1, 3],
                                      [4, 3, 2]])]
        ucart = ucif2ucart(cell_mat_m, u_mats)
        return jnp.trace(ucart[0]) / 3

    def u_equiv_esd(self, var_cov_mat, cell_esd, parameters, cell_par, crystal_system, **kwargs):
        def func(parameters, cell_par):
            cell_mat_m = cell_constants_to_M(*cell_par, crystal_system)
            return self.u_equiv(parameters, cell_mat_m)
        jac1, jac2 = jax.grad(func, [0, 1])(parameters, cell_par)
        esd = jnp.sqrt(
            jac1[None, :] @ var_cov_mat @ jac1[None, :].T 
            + jac2[None,:] @ jnp.diag(cell_esd**2) @ jac2[None,:].T
        )
        return esd

    

@dataclass(frozen=True)
class IsoTFactor(TFactor):
    uiso_value: Value
    adp_type = 'Uiso'
    derived = False

    def as_uij(self, parameters, cell_mat_f, lengths_star, **kwargs):
        uiso = self.uiso_value.resolve(parameters)
        return jnp.array([
            uiso,
            uiso,
            uiso,
            uiso * jnp.sum(cell_mat_f[:, 1] * cell_mat_f[:, 2]) / lengths_star[1] / lengths_star[2],
            uiso * jnp.sum(cell_mat_f[:, 0] * cell_mat_f[:, 2]) / lengths_star[0] / lengths_star[2],
            uiso * jnp.sum(cell_mat_f[:, 0] * cell_mat_f[:, 1]) / lengths_star[0] / lengths_star[1]
        ])

    def uij_esd(self, var_cov_mat):
        return jnp.array([jnp.nan] * 6)

    def u_equiv(self, parameters):
        return self.uiso.resolve(parameters)

    def u_equiv_esd(self, var_cov_mat):
        return self.uiso.resolve_esd(var_cov_mat)

@dataclass(frozen=True)
class UEquivTFactor(TFactor):
    parent_index: int
    scaling_value: Value

    adp_type = 'calc'
    derived = True

    def as_uij(self, uij, cell_mat_m, cell_mat_f, lengths_star, **kwargs):
        uij_parent = uij[self.parent_index, jnp.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])]
        u_cart = ucif2ucart(cell_mat_m, uij_parent[None,:, :])
        uiso = jnp.trace(u_cart[0]) / 3
        return jnp.array([
            uiso,
            uiso,
            uiso,
            uiso * jnp.sum(cell_mat_f[:, 1] * cell_mat_f[:, 2]) / lengths_star[1] / lengths_star[2],
            uiso * jnp.sum(cell_mat_f[:, 0] * cell_mat_f[:, 2]) / lengths_star[0] / lengths_star[2],
            uiso * jnp.sum(cell_mat_f[:, 0] * cell_mat_f[:, 1]) / lengths_star[0] / lengths_star[1]
        ])

    def uij_esd(self, var_cov_mat):
        return jnp.array([jnp.nan] * 6)

    def u_equiv(self, parameters, **kwargs):
        return jnp.nan

    def u_equiv_esd(self, var_cov_mat, **kwargs):
        return jnp.nan

#@dataclass(frozen=True)
#class GCTFactor:
#    gc3_values: Array
#    gc4_values: Array
