""" This module contains the most basic building blocks for structures, which
are not specialised in any way.
"""


import warnings
from collections import namedtuple
from typing import Dict, Tuple, Union
from ..better_abc import ABCMeta, abstract_attribute, abstractmethod
from ..common_jax import jax, jnp
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

class Value(metaclass=ABCMeta):
    """This abstract class represents value classes, i.e. the smallest unit
    translating the parameter vector into the values describing the structure
    """
    @abstractmethod
    def resolve(self, parameters: jnp.array) -> float:
        """Takes in the parameter vector and maps back to a single value

        Parameters
        ----------
        parameters : jnp.array
            This size P array contains the parameter vector, which contains the
            values optimised during refinement

        Returns
        -------
        float
            value calculated from the parameter vector
        """
        pass

    @abstractmethod
    def resolve_esd(self, var_cov_mat: jnp.array) -> float:
        """Takes in the variance covariance matrix and
        gives back the esd of the given calculated value

        Parameters
        ----------
        var_cov_mat : jnp.array
            Size (P,P) array containing the full variance-covariance matrix
            from the least-squares refinement.

        Returns
        -------
        float
            esd for calculated value
        """
        pass

@dataclass(frozen=True)
class RefinedValue(Value):
    """Represents a value that can be calculated from a single value in 
    the parameter vector and therefore changes in the refinement /
    optimisation

    Parameters
    ----------
    par_index : int
        index in the parameter vector that is used as value p in the calculation
    multiplicator : float
        value that the value p is multiplied
    added_value : float
        value that is added to m * p
    """
    par_index: int
    multiplicator: float = 1.0
    added_value: float = 0.0

    def resolve(self, parameters: jnp.array) -> float:
        """Generate the value from the parameter array

        Parameters
        ----------
        parameters : jnp.array
            This size P array contains the parameter vector, which contains the
            values optimised during refinement

        Returns
        -------
        float
            calculated value
        """

        return self.multiplicator * parameters[self.par_index] + self.added_value
    
    def resolve_esd(self, var_cov_mat):
        """Takes in the variance covariance matrix and
        gives back the esd of the given calculated value

        Parameters
        ----------
        var_cov_mat : jnp.array
            Size (P,P) array containing the full variance-covariance matrix
            from the least-squares refinement.

        Returns
        -------
        float
            esd for calculated value
        """
        return jnp.abs(self.multiplicator) * jnp.sqrt(var_cov_mat[self.par_index, self.par_index])

@dataclass(frozen=True)
class FixedValue(Value):
    """Represents a value, which is not refined during refinement / optimisation

    Parameters
    ----------
    value : float
        return value that is used
    """
    value: float

    def resolve(self, parameters: jnp.array) -> float:
        """Give the value determined by this (data) class

        Parameters
        ----------
        parameters : jnp.array
            This size P array contains the parameter vector, which contains the
            values optimised during refinement. Will not be used in a
            FixedValue

        Returns
        -------
        float
            return value as float (given by value attribute)
        """
        return self.value

    def resolve_esd(self, var_cov_mat):
        return jnp.nan # One could pick zero, but this should indicate that an error is not defined

@dataclass(frozen=True)
class MultiIndexValue(Value):
    """Represents a value that can be calculated from multiple values in 
    the parameter vector and therefore changes in the refinement /
    optimisation

    Parameters
    ----------
    par_indexes : Tuple[int]
        The indexes j of the values on the parameter vector 
    multiplicators : Tuple[float]
        the multiplicators which are multiplied with the extracted values
        from the parameter vector m(i) * p(j), has to have the same length
        as par_indexes
    added_value
        single added value that is added to the sum of products
    """
    par_indexes: Tuple[int]
    multiplicators: Tuple[float]
    added_value: float

    def resolve(self, parameters: jnp.array) -> float:
        """Generate the resulting value 

        Parameters
        ----------
        parameters : jnp.array
            This size P array contains the parameter vector, which contains the
            values optimised during refinement. 

        Returns
        -------
        float
            The generated value as float
        """
        multiplicators = jnp.array(self.multiplicators)
        par_values = jnp.take(parameters, jnp.array(self.par_indexes), axis=0)
        return jnp.sum(multiplicators * par_values) + self.added_value
    
    def resolve_esd(self, var_cov_mat):
        """Takes in the variance covariance matrix and
        gives back the esd of the given calculated value

        Parameters
        ----------
        var_cov_mat : jnp.array
            Size (P,P) array containing the full variance-covariance matrix
            from the least-squares refinement.

        Returns
        -------
        float
            esd for calculated value
        """
        jac = jnp.array(self.multiplicators)
        indexes = jnp.array(self.par_indexes)
        return jnp.sqrt(jnp.sqrt(
            jac[None, :] @ var_cov_mat[indexes][: , indexes] @ jac[None, :].T
        ))[0,0]


class AtomicProperty(metaclass=ABCMeta):
    """This ABC represents atomic properties (xyz, uij, occupancy and Gram-
    Charlier) with the exception of thermal parameters

    Parameters
    ----------
    derived : bool
        This property will be calculated in a second cycle where non-derived
        positions and displacement parameters are already available. This 
        is one level higher than Values, which represent the building block,
        which can be used for attributes of AtomicProperties
    """
    derived: bool = abstract_attribute()
    #pre_cycle_calculated: Dict[str, callable] = {}
    #in_cycle_calculated: Dict[str, callable] = {} 

    @abstractmethod
    def resolve(self, parameters: jnp.array, **kwargs) -> jnp.array:
        """Return the values calculated from the parameter array

        Parameters
        ----------
        parameters : jnp.array
            This size P array contains the parameter vector, which contains the
            values optimised during refinement. 

        **kwargs : any other keyword arguments, **kwargs always needs to be
        part of the arguments to deal with arguments that other functions 
        potentially need

        Returns
        -------
        values : jnp.array
            A jax numpy array containing the calculated values
        """
        return 1.0

    @abstractmethod
    def resolve_esd(self, var_cov_mat: jnp.array, **kwargs) -> jnp.array:
        """Return the esd of the values calculated from the parameter array

        Parameters
        ----------
        var_cov_mat : jnp.array
            Size (P,P) array containing the full variance-covariance matrix
            from the least-squares refinement.

        Returns
        -------
        esds
            A jax numpy array containing the calculated esds
        """
        return jnp.nan

@dataclass(frozen=True)
class Array(AtomicProperty):
    """An atomic property that is build by just evaluating a bunch of Value 
    objects from a tuple. Can be used for everything where more attributes
    or more complex resolve functions are not needed.

    Parameters
    ----------
    value_tuple : Tuple[Value]
        A tuple of value objects to evaluate for resolve
    """
    value_tuple: Tuple[Value]
    derived: bool = False
    def resolve(self, parameters: jnp.array, **kwargs) -> jnp.array:
        """Return the values calculated from the parameter array

        Parameters
        ----------
        parameters : jnp.array
            This size P array contains the parameter vector, which contains the
            values optimised during refinement. 

        Returns
        -------
        values : jnp.array
            A jax numpy array containing the calculated values
        """
        return jnp.array(tuple(par.resolve(parameters) for par in self.value_tuple))

    def resolve_esd(self, var_cov_mat: jnp.array, **kwargs) -> jnp.array:
        """Return the esd of the values calculated from the parameter array

        Parameters
        ----------
        var_cov_mat : jnp.array
            Size (P,P) array containing the full variance-covariance matrix
            from the least-squares refinement.

        Returns
        -------
        esds
            A jax numpy array containing the calculated esds
        """
        return jnp.array([par.resolve_esd(var_cov_mat) for par in self.value_tuple])
