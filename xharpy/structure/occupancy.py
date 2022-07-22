"""Module containing the internal object for handling the occupancy"""
from typing import Union
from dataclasses import dataclass
from .common import Value, FixedValue, RefinedValue, AtomicProperty
from ..common_jax import jnp

@dataclass(frozen=True)
class Occupancy(AtomicProperty):
    """This object represents and atomic occupancy.

    Parameters
    ----------
    occ_value: Value
        Value representing the occupancy of this atom
    special_position: bool
        The atom this occupancy belongs to is on a special position. This
        is used for the output to the cif dile
    """
    occ_value: Value
    special_position: bool
    derived: bool = False

    def resolve(
        self,
        parameters: jnp.array
    ) -> float:
        """give the internal occupancy value

        Parameters
        ----------
        parameters : jnp.array
            This size P array contains the parameter vector, which contains the
            values optimised during refinement. 

        Returns
        -------
        occ : float
            occupancy value for atoms on general positions
            occupancy/multiplicity for atoms on special positions
        """
        return self.occ_value.resolve(parameters)

    def resolve_esd(
        self,
        var_cov_mat: jnp.array
    ) -> float:
        """give esd of the the occupancy value

        Parameters
        ----------
        var_cov_mat : jnp.array
            Size (P,P) array containing the full variance-covariance matrix
            from the least-squares refinement.

        Returns
        -------
        float
            esd for occupancy value
        """
        return self.occ_value.resolve_esd(var_cov_mat)

    def occupancy(
        self,
        parameters: jnp.array
    ) -> float:
        """Give the occupancy corrected for multiplicity for cif output

        Parameters
        ----------
        parameters : jnp.array
            This size P array contains the parameter vector, which contains the
            values optimised during refinement. 

        Returns
        -------
        occupancy : float
            Occupancy for atom

        Raises
        ------
        NotImplementedError
            This type of Value object is currently not implemented
        """
        if self.special_position:
            if isinstance(self.occ_value, RefinedValue):
                return self.resolve(parameters) / self.occ_value.multiplicator
            elif isinstance(self.occ_value, FixedValue):
                return 1
            else:
                raise NotImplementedError('Cannot handle this parameters value type')
        else:
            return self.occ_value.resolve(parameters)

    def occupancy_esd(
        self,
        var_cov_mat: jnp.array
    ) -> float:
        """Give the esd of the occupancy corrected for multiplicity for cif
        output

        Parameters
        ----------
        var_cov_mat : jnp.array
            Size (P,P) array containing the full variance-covariance matrix
            from the least-squares refinement.

        Returns
        -------
        float
            Esd of occupancy of atom
        """

        if self.special_position and isinstance(self.occ_value, RefinedValue):
            return self.resolve_esd(var_cov_mat) / self.occ_value.multiplicator
        else:
            return self.occ_value.resolve_esd(var_cov_mat)

    def symmetry_order(self) -> Union[int, float]:
        """Give the symmetry order, 1 for a general position, higher for a
        special position

        Returns
        -------
        symmetry order : Union[int, float]
            The symmetry order of the atom

        Raises
        ------
        NotImplementedError
            This type of Value object is currently not implemented
        """
        if self.special_position: 
            if isinstance(self.occ_value, RefinedValue):
                return int(1 / self.occ_value.multiplicator)
            elif isinstance(self.occ_value, FixedValue):
                return int(1 / self.occ_value.value)
            else:
                raise NotImplementedError('Currently cannot handle this type of variable')
        else:
            return 1
        
