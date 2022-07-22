
"""Contains all the internal constraint objects dealing with positions, that
cannot be expressed as an Array. The objects are implemented as FROZEN 
dataclasses for them to be immutable during refinement."""
from .common import Value, AtomicProperty
from ..common_jax import jnp

from dataclasses import dataclass

@dataclass(frozen=True)
class SingleTrigonalCalculated(AtomicProperty):
    """The internal object for representing a position constraint where the new 
    atom is in plane with three other atoms, of which it is bound to one. Is
    meant for hydrogen atoms bound to sp2 hybridised atoms.

    Parameters
    ----------
    bound_atom_index: int
        index of bound atom
    plane_atom1_index: int
        index of first bonding partner of bound atom
    plane_atom2_index: int
        index of second bonding partner of bound atom
    distance_value: Value
        Value object for generating the distance
    """
    bound_atom_index: int
    plane_atom1_index: int
    plane_atom2_index: int
    distance_value: Value
    derived: bool = True

    def resolve(
        self,
        parameters: jnp.array,
        xyz: jnp.array,
        cell_mat_m: jnp.array,
        **kwargs
    ) -> jnp.array:
        """Calculate the atomic positions, from the already calculated positions
        of the three atoms defining the plane

        Parameters
        ----------
        parameters : jnp.array
            This size P array contains the parameter vector, which contains the
            values optimised during refinement. 
        xyz : jnp.array
            Array containing the precalculated positions of all non-derived
            atoms in the asymmetric unit, must already contain the values
            for the bound_atoms
        cell_mat_m : jnp.ndarray
            size (3, 3) array with the cell vectors as row vectors

        Returns
        -------
        atom_xyz : jnp.array
            size (3) array containing the atomic positions
        """
        bound_xyz = xyz[self.bound_atom_index]
        plane1_xyz = xyz[self.plane_atom1_index]
        plane2_xyz = xyz[self.plane_atom2_index]
        distance = self.distance_value.resolve(parameters)
        direction1 = bound_xyz - plane1_xyz
        direction2 = bound_xyz - plane2_xyz
        addition = (direction1 / jnp.linalg.norm(cell_mat_m @ direction1)
                    + direction2 / jnp.linalg.norm(cell_mat_m @ direction2))
        direction = addition / jnp.linalg.norm(cell_mat_m @ addition)
        return bound_xyz + direction * distance

    def resolve_esd(self, var_cov_mat):
        """Esd of constrained values is always nan
        """
        return jnp.array((jnp.nan, jnp.nan, jnp.nan))

@dataclass(frozen=True)
class TorsionCalculated(AtomicProperty):
    """The internal object for representing constraint where the constraint atom
    is placed by angle, distance and torsion angle to three other atoms.

    Parameters
    ----------
    bound_atom_index: int
        index of atom the derived atom is bound to
    angle_atom_index: int
        index of atom spanning the given angle with bound atom
    torsion_atom_index: int
        index of atom giving the torsion angle
    distance_value: Value
        Value object for generating the distance
    angle_value: Value
        Value object for generating the angle spanned by generated atom,
        bound atom and angle atom
    torsion_angle_value: Value
        Value object for generating the torsion angle spanned by generated atom,
        bound atom and angle atom and torsion atom
    """
    bound_atom_index: int
    angle_atom_index: int
    torsion_atom_index: int
    distance_value: Value
    angle_value: Value
    torsion_angle_value: Value
    derived: bool = True

    def resolve(
        self,
        parameters: jnp.array,
        xyz: jnp.array,
        cell_mat_m: jnp.array,
        cell_mat_f: jnp.array,
        **kwargs
    ) -> jnp.array:
        """Calculate the atomic positions, from the already calculated positions
        of the three atoms defining the torsion angle

        Parameters
        ----------
        parameters : jnp.array
            This size P array contains the parameter vector, which contains the
            values optimised during refinement. 
        xyz : jnp.array
            Array containing the precalculated positions of all non-derived
            atoms in the asymmetric unit, must already contain the values
            for the bound_atoms
        cell_mat_m : jnp.ndarray
            size (3, 3) array with the cell vectors as row vectors
        cell_mat_f : jnp.ndarray
            size (3, 3) array with the reciprocal lattice vectors (1/Angstroem) as row
            vectors

        Returns
        -------
        atom_xyz : jnp.array
            size (3) array containing the atomic positions
        """
        bound_xyz = cell_mat_m @ xyz[self.bound_atom_index]
        angle_xyz = cell_mat_m @ xyz[self.angle_atom_index]
        torsion_xyz = cell_mat_m @ xyz[self.torsion_atom_index]
        distance = self.distance_value.resolve(parameters)
        angle = self.angle_value.resolve(parameters)
        torsion_angle = self.torsion_angle_value.resolve(parameters)
        vec_ab = (angle_xyz - torsion_xyz)
        vec_bc_norm = -(bound_xyz - angle_xyz) / jnp.linalg.norm(bound_xyz - angle_xyz)
        vec_d2 = jnp.array([distance * jnp.cos(angle),
                            distance * jnp.sin(angle) * jnp.cos(torsion_angle),
                            distance * jnp.sin(angle) * jnp.sin(torsion_angle)])
        vec_n = jnp.cross(vec_ab, vec_bc_norm)
        vec_n = vec_n / jnp.linalg.norm(vec_n)
        rotation_mat_m = jnp.array([vec_bc_norm, jnp.cross(vec_n, vec_bc_norm), vec_n]).T
        return cell_mat_f @ (rotation_mat_m @ vec_d2 + bound_xyz)

    def resolve_esd(self, var_cov_mat):
        """Esd of constrained values is always nan
        """
        return jnp.array([jnp.nan] * 3)


@dataclass(frozen=True)
class TetrahedralCalculated(AtomicProperty):
    """The internal object for representing a constraint where the constrained 
    atom completes a tetrahedron with the three tetrahedron atoms around bound 
    atom, by averaging the directions pointing from the tetrahedron atoms to
    bound atom and using that direction to place the new atom with the given
    distance starting at bound_atom. Meant for hydrogen atoms bound to tertiary
    sp3 atoms.

    Parameters
    ----------
    bound_atom_index: int
        index of bound atom
    tetrahedron_atom1_index: int
        index of first atom forming the tetrahedron
    tetrahedron_atom2_index: int
        index of second atom forming the tetrahedron
    tetrahedron_atom3_index: int
        index of third atom forming the tetrahedron
    distance_value: Value
        Value object for generating the distance
    """
    bound_atom_index: int
    tetrahedron_atom1_index: int
    tetrahedron_atom2_index: int
    tetrahedron_atom3_index: int
    distance_value: Value
    derived: bool = True

    def resolve(
        self,
        parameters: jnp.array,
        xyz: jnp.array,
        cell_mat_m: jnp.array,
        **kwargs
    ) -> jnp.array:
        """Calculate the atomic positions, from the already calculated positions
        of the bound atom and the three atoms defining the tetrahedron to be 
        completed.

        Parameters
        ----------
        parameters : jnp.array
            This size P array contains the parameter vector, which contains the
            values optimised during refinement. 
        xyz : jnp.array
            Array containing the precalculated positions of all non-derived
            atoms in the asymmetric unit, must already contain the values
            for the bound_atoms
        cell_mat_m : jnp.ndarray
            size (3, 3) array with the cell vectors as row vectors

        Returns
        -------
        atom_xyz : jnp.array
            size (3) array containing the atomic positions
        """
        bound_xyz = xyz[self.bound_atom_index]
        tetrahedron1_xyz = xyz[self.tetrahedron_atom1_index]
        tetrahedron2_xyz = xyz[self.tetrahedron_atom2_index]
        tetrahedron3_xyz = xyz[self.tetrahedron_atom3_index]
        distance = self.distance_value.resolve(parameters)

        direction1 = bound_xyz - tetrahedron1_xyz
        direction2 = bound_xyz - tetrahedron2_xyz
        direction3 = bound_xyz - tetrahedron3_xyz
        addition = (direction1 / jnp.linalg.norm(cell_mat_m @ direction1)
                    + direction2 / jnp.linalg.norm(cell_mat_m @ direction2)
                    + direction3 / jnp.linalg.norm(cell_mat_m @ direction3))

        direction = (addition) / jnp.linalg.norm(cell_mat_m @ (addition))

        return bound_xyz + direction * distance

    def resolve_esd(self, var_cov_mat):
        """Esd of constrained values is always nan
        """
        return jnp.array([jnp.nan]* 3)
