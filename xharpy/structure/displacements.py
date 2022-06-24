from .common import Parameter, Array

from .common import jnp
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

@dataclass(frozen=True)
class AnisoTFactor:
    uij_pars: Array

    def as_uij(self, parameters, **kwargs):
        return self.uij_pars.resolve(parameters)

    def uij_esd(self, var_cov_mat):
        return self.uij_pars.resolve_esd(var_cov_mat)

    def u_equiv(self, parameters):
        pass

    def u_equiv_esd(self, var_cov_mat):
        pass

@dataclass(frozen=True)
class IsoTFactor:
    uiso_par: Parameter

    def as_uij(self, parameters, cell_mat_f, lengths_star):
        uiso = self.uiso_par.resolve(parameters)
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
class UEquivTFactor:
    parent_index: int
    scaling_par: Parameter

    def as_uij(self, uij, cell_mat_m, cell_mat_f, lengths_star):
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

#@dataclass(frozen=True)
#class GCTFactor:
#    gc3_pars: Array
#    gc4_pars: Array
