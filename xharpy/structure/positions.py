
from .common import Parameter
from .common import jnp

from dataclasses import dataclass

@dataclass(frozen=True)
class SingleTrigonalCalculated:
    bound_atom_index: int
    plane_atom1_index: int
    plane_atom2_index: int
    distance_par: Parameter
    derived: bool = True

    def resolve(self, parameters, xyz, cell_mat_m, **kwargs):
        bound_xyz = xyz[self.bound_atom_index]
        plane1_xyz = xyz[self.plane_atom1_index]
        plane2_xyz = xyz[self.plane_atom2_index]
        distance = self.distance_par.resolve(parameters)
        direction1 = bound_xyz - plane1_xyz
        direction2 = bound_xyz - plane2_xyz
        addition = (direction1 / jnp.linalg.norm(cell_mat_m @ direction1)
                    + direction2 / jnp.linalg.norm(cell_mat_m @ direction2))
        direction = addition / jnp.linalg.norm(cell_mat_m @ addition)
        return bound_xyz + direction * distance

    def resolve_esd(self, var_cov_mat):
        return jnp.array((jnp.nan, jnp.nan, jnp.nan))

@dataclass(frozen=True)
class TorsionCalculated:
    bound_atom_index: int
    angle_atom_index: int
    torsion_atom_index: int
    distance_par: Parameter
    angle_par: Parameter
    torsion_angle_par: Parameter
    derived: bool = True

    def resolve(self, parameters, xyz, cell_mat_m, cell_mat_f, **kwargs):
        bound_xyz = cell_mat_m @ xyz[self.bound_atom_index]
        angle_xyz = cell_mat_m @ xyz[self.angle_atom_index]
        torsion_xyz = cell_mat_m @ xyz[self.torsion_atom_index]
        distance = self.distance_par.resolve(parameters)
        angle = self.angle_par.resolve(parameters)
        torsion_angle = self.torsion_angle_par.resolve(parameters)
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
        return jnp.array([jnp.nan] * 3)


@dataclass(frozen=True)
class TetrahedralCalculated:
    bound_atom_index: int
    tetrahedron_atom1_index: int
    tetrahedron_atom2_index: int
    tetrahedron_atom3_index: int
    distance_par: Parameter
    derived: bool = True

    def resolve(self, parameters, xyz, cell_mat_m, **kwargs):
        bound_xyz = xyz[self.bound_atom_index]
        tetrahedron1_xyz = xyz[self.tetrahedron_atom1_index]
        tetrahedron2_xyz = xyz[self.tetrahedron_atom2_index]
        tetrahedron3_xyz = xyz[self.tetrahedron_atom3_index]
        distance = self.distance_par.resolve(parameters)

        direction1 = bound_xyz - tetrahedron1_xyz
        direction2 = bound_xyz - tetrahedron2_xyz
        direction3 = bound_xyz - tetrahedron3_xyz
        addition = (direction1 / jnp.linalg.norm(cell_mat_m @ direction1)
                    + direction2 / jnp.linalg.norm(cell_mat_m @ direction2)
                    + direction3 / jnp.linalg.norm(cell_mat_m @ direction3))

        direction = (addition) / jnp.linalg.norm(cell_mat_m @ (addition))

        return bound_xyz + direction * distance

    def resolve_esd(self, var_cov_mat):
        return jnp.array([jnp.nan]* 3)
