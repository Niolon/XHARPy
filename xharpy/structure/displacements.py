""" This module contains all the objects in use for the description of 
displacement parameters. Currently all of these objects map back to uij, but 
in the future they should calculate the temperature factor directly from 
index_vec_h
"""

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
    """This base class represents calculationgs involving displacement

    Parameters
    ----------
    adp_type : str
        ADP type as given in the atom table in the cif file
    derived : bool
        This property will be calculated in a second cycle where non-derived
        positions and displacement parameters are already available. This 
        is one level higher than Values, which represent the building block,
        which can be used for attributes of AtomicProperties
    """
    adp_type: str = abstract_attribute()
    derived: bool = abstract_attribute()

    @abstractmethod
    def as_uij(self, parameters: jnp.array, **kwargs) -> jnp.array:
        """Calculate the equivalent uij values from the parameter vector

        Parameters
        ----------
        parameters : jnp.array
            This size P array contains the parameter vector, which contains the
            values optimised during refinement. 

        Returns
        -------
        uij : jnp.array
            Size (6) array that contains the uij parameters in the order
            U11, U22, U33, U23, U13, U12
        """
        return None
    
    @abstractmethod
    def uij_esd(self, var_cov_mat: jnp.array, **kwargs) -> jnp.array:
        """Calculate the esds of the equivalent uij values from the variance
        covariance matrix. Can also return jnp.nan values if esds are not 
        usually calculated (e.g. Uiso)

        Parameters
        ----------
        var_cov_mat : jnp.array
            Size (P,P) array containing the full variance-covariance matrix
            from the least-squares refinement.

        Returns
        -------
        uij_esd : jnp.array
            Size (6) array that contains the esd of the uij parameters in the 
            order U11, U22, U33, U23, U13, U12
        """
        return None

    @abstractmethod
    def u_equiv(self, parameters: jnp.array, **kwargs) -> float:
        """Returns the U(equiv) value corresponding to the calculated 
        displacement parameters

        Parameters
        ----------
        parameters : jnp.array
            This size P array contains the parameter vector, which contains the
            values optimised during refinement.
 
        Returns
        -------
        u_equiv : float
            the calculated u_equiv value
        """
        return None

    @abstractmethod
    def u_equiv_esd(self, var_cov_mat: jnp.array, **kwargs) -> float:
        """Returns the esd of the U(equiv) value corresponding to the calculated 
        displacement parameters

        Parameters
        ----------
        var_cov_mat : jnp.array
            Size (P,P) array containing the full variance-covariance matrix
            from the least-squares refinement.

        Returns
        -------
        u_equiv_esd : float
            the calculated esd of the u_equiv value
        """
        return None


@dataclass(frozen=True)
class AnisoTFactor(TFactor):
    """Represents the calculation from anisotropic displacement parameters in 
    in the cif convention

    Parameters
    ----------
    uij_values : Array
        Array containing the values that represent the uij values
    """
    uij_values: Array
    adp_type = 'Uani'
    derived = False

    def as_uij(self, parameters: jnp.array, **kwargs) -> jnp.array:
        """Calculate the equivalent uij values from the parameter vector

        Parameters
        ----------
        parameters : jnp.array
            This size P array contains the parameter vector, which contains the
            values optimised during refinement. 

        Returns
        -------
        uij : jnp.array
            Size (6) array that contains the uij parameters in the order
            U11, U22, U33, U23, U13, U12
        """
        return self.uij_values.resolve(parameters)

    def uij_esd(self, var_cov_mat):
        """Calculate the esds of the equivalent uij values from the variance
        covariance matrix. Can also return jnp.nan values if esds are not 
        usually calculated (e.g. Uiso)

        Parameters
        ----------
        var_cov_mat : jnp.array
            Size (P,P) array containing the full variance-covariance matrix
            from the least-squares refinement.

        Returns
        -------
        uij_esd : jnp.array
            Size (6) array that contains the esd of the uij parameters in the 
            order U11, U22, U33, U23, U13, U12
        """
        return self.uij_values.resolve_esd(var_cov_mat)

    def u_equiv(
        self,
        parameters: jnp.array,
        cell_mat_m: jnp.array,
        **kwargs
        ) -> float:
        """Returns the U(equiv) value corresponding to the calculated 
        displacement parameters

        Parameters
        ----------
        parameters : jnp.array
            This size P array contains the parameter vector, which contains the
            values optimised during refinement. 
        cell_mat_m : jnp.array
            size (3, 3) array with the cell vectors as row vectors (Angstroem)
        
        Returns
        -------
        u_equiv : float
            the calculated u_equiv value
        """
        uij = self.uij_values.resolve(parameters)
        u_mats = uij[None, jnp.array([[0, 5, 4],
                                      [5, 1, 3],
                                      [4, 3, 2]])]
        ucart = ucif2ucart(cell_mat_m, u_mats)
        return jnp.trace(ucart[0]) / 3

    def u_equiv_esd(
        self,
        var_cov_mat : jnp.array,
        cell_esd : jnp.array,
        parameters : jnp.array,
        cell_par: jnp.array,
        crystal_system : jnp.array,
        **kwargs
        ):
        """eturns the esd of the U(equiv) value corresponding to the calculated 
        displacement parameters

        Parameters
        ----------
        var_cov_mat : jnp.array
            Size (P,P) array containing the full variance-covariance matrix
            from the least-squares refinement.
        cell_esd : jnp.array
            array with the estimated standard deviation of the lattice constants
            (Angstroem, Degree)
        parameters : jnp.array
            final refined parameters
        cell_par : jnp.array
            array with the lattice constants (Angstroem, Degree)
        crystal_system : jnp.array
            Crystal system of the evaluated structure. Possible values are: 
            'triclinic', 'monoclinic' 'orthorhombic', 'tetragonal', 'hexagonal',
            'trigonal' and 'cubic'.
        """
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
    """Represents an isotropic refinement of displacement for the given atom

    Parameters
    ----------
    uiso_value : Value
        Value object representing the u(iso) value
    """
    uiso_value: Value
    adp_type = 'Uiso'
    derived = False

    def as_uij(
        self,
        parameters: jnp.array,
        cell_mat_f: jnp.array,
        lengths_star: jnp.array,
        **kwargs
    ) -> jnp.array:
        """Calculate the equivalent uij values from the parameter vector

        Parameters
        ----------
        parameters : jnp.array
            This size P array contains the parameter vector, which contains the
            values optimised during refinement. 
        cell_mat_f : jnp.ndarray
            size (3, 3) array with the reciprocal lattice vectors (1/Angstroem)
            as row vectors
        lengths_star : jnp.ndarray
            size (3) array with the lengths of the reciprocal lattice vectors 
            (1/Angstroem)

        Returns
        -------
        uij : jnp.array
            Size (6) array that contains the uij parameters in the order
            U11, U22, U33, U23, U13, U12
        """
        uiso = self.uiso_value.resolve(parameters)
        return jnp.array([
            uiso,
            uiso,
            uiso,
            uiso * jnp.sum(cell_mat_f[:, 1] * cell_mat_f[:, 2]) / lengths_star[1] / lengths_star[2],
            uiso * jnp.sum(cell_mat_f[:, 0] * cell_mat_f[:, 2]) / lengths_star[0] / lengths_star[2],
            uiso * jnp.sum(cell_mat_f[:, 0] * cell_mat_f[:, 1]) / lengths_star[0] / lengths_star[1]
        ])

    def uij_esd(self, var_cov_mat, **kwargs):
        """Returns a size 6 array filled with nan values, as the uij esd of 
        an isotropic atom should not be used on equal footing with refined uij
        values
        """
        return jnp.array([jnp.nan] * 6)

    def u_equiv(self, parameters):
        """Returns the U(equiv)/U(iso) value

        Parameters
        ----------
        parameters : jnp.array
            This size P array contains the parameter vector, which contains the
            values optimised during refinement. 
        Returns
        -------
        u_equiv : float
            the calculated u_equiv value
        """
        return self.uiso.resolve(parameters)

    def u_equiv_esd(self, var_cov_mat):
        """Returns the esd of the U(equiv) value corresponding to the calculated 
        displacement parameters

        Parameters
        ----------
        var_cov_mat : jnp.array
            Size (P,P) array containing the full variance-covariance matrix
            from the least-squares refinement.

        Returns
        -------
        u_equiv_esd : float
            the calculated esd of the u_equiv value
        """
        return self.uiso.resolve_esd(var_cov_mat)

@dataclass(frozen=True)
class UEquivTFactor(TFactor):
    """Represents the case, where the displacement of one atom is calculated as
    as a multiple of Uequiv of another atom

    Parameters
    ----------
    parent_index : int
        Index of the atom from which the Uequiv value is derived
    scaling_value : Value
        Value representing the scaling between the two atoms

    """
    parent_index: int
    scaling_value: Value

    adp_type = 'calc'
    derived = True

    def as_uij(
        self,
        uij: jnp.array,
        cell_mat_m: jnp.array,
        cell_mat_f: jnp.array,
        lengths_star: jnp.array,
        **kwargs
    ) -> jnp.array:
        """Calculate the equivalent uij values. Values are calculated from
        the parent uij values

        Parameters
        ----------
        uij : jnp.array
            This array contains the also calculated (non-derived) uij values
        cell_mat_m : jnp.ndarray
            size (3, 3) array with the cell vectors as row vectors
         cell_mat_f : jnp.ndarray
            size (3, 3) array with the reciprocal lattice vectors (1/Angstroem)
            as row vectors
        lengths_star : jnp.ndarray
            size (3) array with the lengths of the reciprocal lattice vectors 
            (1/Angstroem)       

        Returns
        -------
        uij : jnp.array
            Size (6) array that contains the uij parameters in the order
            U11, U22, U33, U23, U13, U12
        """
        uiso = self.u_equiv(uij, cell_mat_m)
        return jnp.array([
            uiso,
            uiso,
            uiso,
            uiso * jnp.sum(cell_mat_f[:, 1] * cell_mat_f[:, 2]) / lengths_star[1] / lengths_star[2],
            uiso * jnp.sum(cell_mat_f[:, 0] * cell_mat_f[:, 2]) / lengths_star[0] / lengths_star[2],
            uiso * jnp.sum(cell_mat_f[:, 0] * cell_mat_f[:, 1]) / lengths_star[0] / lengths_star[1]
        ])

    def uij_esd(self, var_cov_mat):
        """Returns a size 6 array filled with nan values, as the uij esd of 
        an isotropic atom should not be used on equal footing with refined uij
        values
        """
        return jnp.array([jnp.nan] * 6)

    def u_equiv(self, uij, cell_mat_m, **kwargs):
        """Returns the U(equiv) value corresponding to the parent 
        displacement parameters

        Parameters
        ----------
        uij : jnp.array
            This array contains the also calculated (non-derived) uij values
        cell_mat_m : jnp.ndarray
            size (3, 3) array with the cell vectors as row vectors
        
        Returns
        -------
        u_equiv : float
            the calculated u_equiv value
        """
        indexes = jnp.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])
        uij_parent = uij[self.parent_index, indexes]
        u_cart = ucif2ucart(cell_mat_m, uij_parent[None,:, :])
        uiso = jnp.trace(u_cart[0]) / 3
        return uiso

    def u_equiv_esd(self, var_cov_mat, **kwargs):
        """Returns nan as the esd of this value should not be compared on equal
        footing with other esds
        """
        return jnp.nan

#@dataclass(frozen=True)
#class GCTFactor:
#    gc3_values: Array
#    gc4_values: Array
