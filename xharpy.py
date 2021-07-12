from jax.config import config
config.update('jax_enable_x64', True)

import jax.numpy as jnp
import pandas as pd
import os
from collections import OrderedDict, namedtuple
from itertools import product
import warnings
import jax
import numpy as np
import pickle
from scipy.optimize import minimize
from jax.scipy.optimize import minimize as jminimize

from .restraints import resolve_restraints
from .conversion import ucif2ucart, cell_constants_to_M


def expand_symm_unique(type_symbols, coordinates, cell_mat_m, symm_mats_vec, skip_symm={}):
    """Expand the type_symbols and coordinates for one complete unit cell
    Will return an atom coordinate on a special position only once 
    also returns the matrix inv_indexes with shape n_symm * n_at for
    reconstructing the complete atom list including multiples of special position atoms"""
    symm_mats_r, symm_vecs_t = symm_mats_vec
    pos_frac0 = coordinates % 1
    un_positions = np.zeros((0, 3))
    n_atoms = 0
    type_symbols_symm = []
    inv_indexes = []
    # Only check atom with itself
    for atom_index, (pos0, type_symbol) in enumerate(zip(pos_frac0, type_symbols)):
        if atom_index in skip_symm:
            use_indexes = [i for i in range(symm_mats_r.shape[0]) if i not in skip_symm[atom_index]]
        else:
            use_indexes = list(range(symm_mats_r.shape[0]))
        symm_positions = (np.einsum('kxy, y -> kx', symm_mats_r[use_indexes, :, :], pos0) + symm_vecs_t[use_indexes, :]) % 1
        _, unique_indexes, inv_indexes_at = np.unique(np.round(np.einsum('xy, zy -> zx', cell_mat_m, symm_positions), 3),
                                                      axis=0,
                                                      return_index=True,
                                                      return_inverse=True)
        un_positions = np.concatenate((un_positions, symm_positions[unique_indexes]))
        type_symbols_symm += [type_symbol] * unique_indexes.shape[0]
        inv_indexes.append(inv_indexes_at + n_atoms)
        n_atoms += unique_indexes.shape[0]
    return un_positions.copy(), type_symbols_symm, inv_indexes


@jax.jit
def calc_f(xyz, uij, cijk, dijkl, occupancies, index_vec_h, cell_mat_f, symm_mats_vecs, fjs):
    """Calculate the overall structure factors for given indexes of hkl"""
    
    #einsum indexes: k: n_symm, z: n_atom, h: n_hkl
    lengths_star = jnp.linalg.norm(cell_mat_f, axis=0)
    #cell_mat_g_star = jnp.einsum('ja, jb -> ab', cell_mat_f, cell_mat_f)
    symm_mats_r, symm_vecs_t = symm_mats_vecs
    #vec_S = jnp.einsum('xy, zy -> zx', cell_mat_f, index_vec_h)
    #vec_S_symm = jnp.einsum('kxy, zy -> kzx', symm_mats_r, vec_S)
    #vec_S_symm = jnp.einsum('zx, kxy -> kzy', vec_S, symm_mats_r) # entspricht H.T @ R
    vec_h_symm = jnp.einsum('zx, kxy -> kzy', index_vec_h, symm_mats_r) # entspricht H.T @ R
    u_mats = uij[:, jnp.array([[0, 5, 4],
                              [5, 1, 3],
                              [4, 3, 2]])]
    vib_factors = jnp.exp(-2 * jnp.pi**2 * jnp.einsum('kha, khb, zab -> kzh', vec_h_symm, vec_h_symm, u_mats * jnp.outer(lengths_star, lengths_star)))
    #vib_factors = jnp.exp(-2 * jnp.pi**2 * jnp.einsum('kha, khb, zab -> kzh', vec_h_symm, vec_h_symm, u_mats * cell_mat_g_star))
    
    #vib_factors = jnp.exp(-2 * jnp.pi**2 * jnp.einsum('kha, khb, zab -> kzh', vec_S_symm, vec_S_symm, u_mats))
    #TODO Check if Gram-Charlier is correct this way
    gram_charlier3_indexes = jnp.array([[[0, 3, 5], [3, 4, 9], [5, 9, 6]],
                                        [[3, 4, 9], [4, 1, 7], [9, 7, 8]],
                                        [[5, 9, 6], [9, 7, 8], [6, 8, 2]]])
    cijk_inner_sum = jnp.einsum('kha, khb, khc, zabc -> kzh',
                               vec_h_symm,
                               vec_h_symm,
                               vec_h_symm,
                               cijk[:, gram_charlier3_indexes])
    gram_charlier3 = (4.0j * jnp.pi**3 / 3) * cijk_inner_sum
    gram_charlier4_indexes = jnp.array([[[[0, 3, 5],    [3, 9, 12],   [5, 12, 10]],
                                         [[3, 9, 12],   [9, 4, 13],  [12, 13, 14]],
                                         [[5, 12, 10], [12, 13, 14], [10, 14, 6]]],
                                        [[[3, 9, 12],  [9, 4, 13],   [12, 13, 14]],
                                         [[9, 4, 13],   [4, 1, 7],   [13, 7, 11]],
                                         [[12, 13, 14], [13, 7, 11], [14, 11, 8]]],
                                        [[[5, 12, 10], [12, 13, 14], [10, 14, 6]],
                                         [[12, 13, 14], [13, 7, 11], [14, 11, 8]],
                                         [[10, 14, 6], [14, 11, 8], [6, 8, 2]]]])
    dijkl_inner_sum = jnp.einsum('kha, khb, khc, khd, zabcd -> kzh',
                                 vec_h_symm,
                                 vec_h_symm,
                                 vec_h_symm,
                                 vec_h_symm,
                                 dijkl[:, gram_charlier4_indexes])
    gram_charlier4 = (2.0 * jnp.pi**2 / 3.0) * dijkl_inner_sum
    gc_factor = 1 - gram_charlier3 + gram_charlier4

    positions_symm = jnp.einsum('kxy, zy -> kzx', symm_mats_r, xyz) + symm_vecs_t[:, None, :]
    phases = jnp.exp(2j * jnp.pi * jnp.einsum('kzx, hx -> kzh', positions_symm, index_vec_h))
    structure_factors = jnp.sum(occupancies[None, :] *  jnp.einsum('kzh, kzh, kzh, kzh -> hz', phases, vib_factors, fjs, gc_factor), axis=-1)
    return structure_factors


def resolve_instruction(parameters, instruction):
    """Resolve fixed and refined parameters"""
    if type(instruction).__name__ == 'RefinedParameter':
        return_value = instruction.multiplicator * parameters[instruction.par_index] + instruction.added_value
    elif type(instruction).__name__ == 'FixedParameter':
        return_value = instruction.value
    else:
        raise NotImplementedError('This type of instruction is not implemented')
    return return_value


# construct the instructions for building the atomic parameters back from the linear parameter matrix
def create_construction_instructions(atom_table, constraint_dict, sp2_add, torsion_add, atoms_for_gc3, atoms_for_gc4, scaling0=1.0, exti0=0.0, refinement_dict={}):
    """
    Creates the instructions that are needed for reconstructing all atomic parameters from the refined parameters
    Additionally returns an initial guesss for the refined parameter list from the atom table.
    """
    parameters = jnp.full(10000, jnp.nan)
    current_index = 1
    parameters = jax.ops.index_update(parameters, jax.ops.index[0], scaling0)
    if 'flack' in refinement_dict:
        if refinement_dict['flack']:
            parameters = jax.ops.index_update(parameters, jax.ops.index[current_index], 0.0)
            current_index += 1
    if 'core' in refinement_dict:
        if refinement_dict['core'] == 'scale':
            parameters = jax.ops.index_update(parameters, jax.ops.index[current_index], 1.0)
            current_index += 1
    if 'extinction' in refinement_dict:
        if refinement_dict['extinction'] != 'none':
            parameters = jax.ops.index_update(parameters, jax.ops.index[current_index], exti0)
            current_index += 1            
    construction_instructions = []
    known_torsion_indexes = {}
    names = atom_table['label']   
    for _, atom in atom_table.iterrows():
        if atom['label'] in sp2_add.keys():
            bound_atom, plane_atom1, plane_atom2, distance, occupancy = sp2_add[atom['label']]
            bound_index = jnp.where(names == bound_atom)[0][0]
            plane_atom1_index = jnp.where(names == plane_atom1)[0][0]
            plane_atom2_index = jnp.where(names == plane_atom2)[0][0]
            xyz_constraint = SingleTrigonalCalculated(bound_atom_index=bound_index,
                                                      plane_atom1_index=plane_atom1_index,
                                                      plane_atom2_index=plane_atom2_index,
                                                      distance=distance)
            adp_constraint = UEquivCalculated(atom_index=bound_index,
                                              multiplicator=1.2)
            construction_instructions.append(AtomInstructions(xyz=xyz_constraint,
                                                              uij=adp_constraint,
                                                              occupancy=FixedParameter(value=float(occupancy))))
            continue
        if atom['label'] in torsion_add.keys():
            bound_atom, angle_atom, torsion_atom, distance, angle, torsion_angle_add, group_index, occupancy = torsion_add[atom['label']]
            if group_index not in known_torsion_indexes.keys():
                known_torsion_indexes[group_index] = current_index
                #TODO: The torsion add_start needs to be calculated for each group
                torsion_add_start = 0.0
                parameters = jax.ops.index_update(parameters, jax.ops.index[current_index], torsion_add_start)
                current_index += 1
            bound_index = jnp.where(names == bound_atom)[0][0]
            angle_index = jnp.where(names == angle_atom)[0][0]
            torsion_index = jnp.where(names == torsion_atom)[0][0]
            torsion_parameter = RefinedParameter(par_index=known_torsion_indexes[group_index],
                                                 multiplicator=1.0,
                                                 added_value=torsion_angle_add)
            xyz_constraint = TorsionCalculated(bound_atom_index=bound_index,
                                               angle_atom_index=angle_index,
                                               torsion_atom_index=torsion_index,
                                               distance=FixedParameter(value=float(distance)),
                                               angle=FixedParameter(value=float(angle)),
                                               torsion_angle=torsion_parameter)
            adp_constraint = UEquivCalculated(atom_index=bound_index,
                                              multiplicator=1.5)
            construction_instructions.append(AtomInstructions(xyz=xyz_constraint,
                                                              uij=adp_constraint,
                                                              occupancy=FixedParameter(value=float(occupancy))))
            continue

        xyz = atom[['fract_x', 'fract_y', 'fract_z']].values.astype(np.float64)
        if atom['label'] in constraint_dict.keys() and 'xyz' in constraint_dict[atom['label']].keys():
            constraint = constraint_dict[atom['label']]['xyz']
            instr_zip = zip(constraint.variable_indexes, constraint.multiplicators, constraint.added_value)
            xyz_instructions = tuple(RefinedParameter(par_index=int(current_index + par_index),
                                                      multiplicator=mult,
                                                      added_value=add) if par_index >= 0 
                                     else FixedParameter(value=float(add), special_position=constraint.special_position) for par_index, mult, add in instr_zip)
            n_pars = jnp.max(constraint.variable_indexes) + 1
            parameters = jax.ops.index_update(parameters, jax.ops.index[current_index:current_index + n_pars],
                                              [xyz[jnp.array(varindex)] for index, varindex in zip(*np.unique(constraint.variable_indexes, return_index=True)) if index >=0])
            current_index += n_pars
        else:
            parameters = jax.ops.index_update(parameters, jax.ops.index[current_index:current_index + 3], list(xyz))
            xyz_instructions = tuple(RefinedParameter(par_index=int(array_index), multiplicator=1.0) for array_index in range(current_index, current_index + 3))
            current_index += 3

        if atom['adp_type'] == 'Uani' or atom['adp_type'] == 'Umpe':
            adp = jnp.array(atom[['U_11', 'U_22', 'U_33', 'U_23', 'U_13', 'U_12']].values.astype(np.float64))
            if atom['label'] in constraint_dict.keys() and 'uij' in constraint_dict[atom['label']].keys():
                constraint = constraint_dict[atom['label']]['uij']
                if type(constraint).__name__ == 'ConstrainedValues':
                    instr_zip = zip(constraint.variable_indexes, constraint.multiplicators, constraint.added_value) 
                    adp_instructions = tuple(RefinedParameter(par_index=int(current_index + par_index),
                                                            multiplicator= mult,
                                                            added_value=add) if par_index >= 0 
                                            else FixedParameter(value=float(add), special_position=constraint.special_position) for par_index, mult, add in instr_zip)
                    n_pars = jnp.max(constraint.variable_indexes) + 1

                    parameters = jax.ops.index_update(parameters, jax.ops.index[current_index:current_index + n_pars],
                                                    [adp[jnp.array(varindex)] for index, varindex in zip(*np.unique(constraint.variable_indexes, return_index=True)) if index >=0])
                    current_index += n_pars
                elif type(constraint).__name__ == 'UEquivConstraint':
                    bound_index = jnp.where(names == constraint.bound_atom)[0][0]
                    adp_instructions = UEquivCalculated(atom_index=bound_index, multiplicator=constraint.multiplicator)
                else:
                    raise NotImplementedError('Unknown Uij Constraint')
            else:
                parameters = jax.ops.index_update(parameters, jax.ops.index[current_index:current_index + 6], list(adp))
                adp_instructions = tuple(RefinedParameter(par_index=int(array_index), multiplicator=1.0) for array_index in range(current_index, current_index + 6))
                current_index += 6
        elif atom['adp_type'] == 'Uiso':
            adp_instructions = UIso(uiso=RefinedParameter(par_index=int(current_index), multiplicator=1.0, added_value=0.0))
            parameters = jax.ops.index_update(parameters, jax.ops.index[current_index], atom['U_iso_or_equiv'])
            current_index += 1
        else:
            raise NotImplementedError('Unknown ADP type in cif. Please use the Uiso or Uani convention')

        if 'C_111' in atom.keys():
            cijk = jnp.array(atom[['C_111', 'C_222', 'C_333', 'C_112', 'C_122', 'C_113', 'C_133', 'C_223', 'C_233', 'C_123']].values.astype(np.float64))
        else:
            cijk = jnp.zeros(10)

        if atom['label'] in constraint_dict.keys() and 'cijk' in constraint_dict[atom['label']].keys() and atom['label'] in atoms_for_gc3:
            constraint = constraint_dict[atom['label']]['cijk']
            instr_zip = zip(constraint.variable_indexes, constraint.multiplicators, constraint.added_value) 
            cijk_instructions = tuple(RefinedParameter(par_index=int(current_index + par_index),
                                                       multiplicator=mult,
                                                       added_value=add) if par_index >= 0 
                                      else FixedParameter(value=float(add), special_position=constraint.special_position) for par_index, mult, add in instr_zip)
            n_pars = jnp.max(constraint.variable_indexes) + 1

            parameters = jax.ops.index_update(parameters, jax.ops.index[current_index:current_index + n_pars],
                                              [cijk[jnp.array(varindex)] for index, varindex in zip(*np.unique(constraint.variable_indexes, return_index=True)) if index >=0])
            current_index += n_pars
        elif atom['label'] in atoms_for_gc3:
            parameters = jax.ops.index_update(parameters, jax.ops.index[current_index:current_index + 10], list(cijk))
            cijk_instructions = tuple(RefinedParameter(par_index=int(array_index), multiplicator=1.0) for array_index in range(current_index, current_index + 10))
            current_index += 10
        else:
            cijk_instructions = tuple(FixedParameter(value=0.0) for index in range(10))

        if 'D_1111' in atom.keys():
            dijkl = jnp.array(atom[['D_1111', 'D_2222', 'D_3333', 'D_1112', 'D_1222', 'D_1113', 'D_1333', 'D_2223', 'D_2333', 'D_1122', 'D_1133', 'D_2233', 'D_1123', 'D_1223', 'D_1233']].values.astype(np.float64))
        else:
            dijkl = jnp.zeros(15)

        if atom['label'] in constraint_dict.keys() and 'dijkl' in constraint_dict[atom['label']].keys() and atom['label'] in atoms_for_gc4:
            constraint = constraint_dict[atom['label']]['dijkl']
            instr_zip = zip(constraint.variable_indexes, constraint.multiplicators, constraint.added_value) 
            dijkl_instructions = tuple(RefinedParameter(par_index=int(current_index + par_index),
                                                        multiplicator=mult,
                                                        added_value=add) if par_index >= 0 
                                      else FixedParameter(value=float(add), special_position=constraint.special_position) for par_index, mult, add in instr_zip)
            n_pars = jnp.max(constraint.variable_indexes) + 1

            parameters = jax.ops.index_update(parameters, jax.ops.index[current_index:current_index + n_pars],
                                              [dijkl[jnp.array(varindex)] for index, varindex in zip(*np.unique(constraint.variable_indexes, return_index=True)) if index >=0])
            current_index += n_pars
        elif atom['label'] in atoms_for_gc4:
            parameters = jax.ops.index_update(parameters, jax.ops.index[current_index:current_index + 15], list(dijkl))
            dijkl_instructions = tuple(RefinedParameter(par_index=int(array_index), multiplicator=1.0) for array_index in range(current_index, current_index + 15))
            current_index += 15
        else:
            dijkl_instructions = tuple(FixedParameter(value=0.0) for index in range(15))

        if atom['label'] in constraint_dict.keys() and 'occ' in constraint_dict[atom['label']].keys():
            constraint = constraint_dict[atom['label']]['occ']
            occupancy = FixedParameter(value=float(constraint.added_value[0]), special_position=constraint.special_position)
        else:
            occupancy = FixedParameter(value=float(atom['occupancy']))

        construction_instructions.append(AtomInstructions(
            name=atom['label'],
            element=atom['type_symbol'],
            dispersion_real = atom['type_scat_dispersion_real'],
            dispersion_imag = atom['type_scat_dispersion_imag'],
            xyz=xyz_instructions,
            uij=adp_instructions,
            cijk=cijk_instructions,
            dijkl=dijkl_instructions,
            occupancy=occupancy
        ))
    parameters = parameters[:current_index]
    return tuple(construction_instructions), parameters



def construct_values(parameters, construction_instructions, cell_mat_m):
    """Reconstruct xyz, adp-parameters and occupancies from the given construction instructions. Allows for the
    Flexible usage of combinations of fixed parameters and parameters that are refined, as well as constraints"""
    cell_mat_f = jnp.linalg.inv(cell_mat_m).T
    lengths_star = jnp.linalg.norm(cell_mat_f, axis=0)
    xyz = jnp.array(
        [jnp.array([resolve_instruction(parameters, inner_instruction) for inner_instruction in instruction.xyz])
          if type(instruction.xyz) in (tuple, list) else jnp.full(3, -9999.9) for instruction in construction_instructions]
    )
    uij = jnp.array(
        [jnp.array([resolve_instruction(parameters, inner_instruction) for inner_instruction in instruction.uij])
          if type(instruction.uij) in (tuple, list) else jnp.full(6, -9999.9) for instruction in construction_instructions]
    )
    
    cijk = jnp.array(
        [jnp.array([resolve_instruction(parameters, inner_instruction) for inner_instruction in instruction.cijk])
          if type(instruction.cijk) in (tuple, list) else jnp.full(6, -9999.9) for instruction in construction_instructions]
    )
    
    dijkl = jnp.array(
        [jnp.array([resolve_instruction(parameters, inner_instruction) for inner_instruction in instruction.dijkl])
          if type(instruction.dijkl) in (tuple, list) else jnp.full(6, -9999.9) for instruction in construction_instructions]
    )
    occupancies = jnp.array([resolve_instruction(parameters, instruction.occupancy) for instruction in construction_instructions])    

    # second loop here for constructed options in order to have everything already available
    for index, instruction in enumerate(construction_instructions):
        # constrained positions
        if type(instruction.xyz).__name__ == 'TorsionCalculated':
            bound_xyz = cell_mat_m @ xyz[instruction.xyz.bound_atom_index]
            angle_xyz = cell_mat_m @ xyz[instruction.xyz.angle_atom_index]
            torsion_xyz = cell_mat_m @ xyz[instruction.xyz.torsion_atom_index]
            vec_ab = (angle_xyz - torsion_xyz)
            vec_bc_norm = -(bound_xyz - angle_xyz) / jnp.linalg.norm(bound_xyz - angle_xyz)
            distance = resolve_instruction(parameters, instruction.xyz.distance)
            angle = jnp.deg2rad(resolve_instruction(parameters, instruction.xyz.angle))
            torsion_angle = jnp.deg2rad(resolve_instruction(parameters, instruction.xyz.torsion_angle))
            vec_d2 = jnp.array([distance * jnp.cos(angle),
                                distance * jnp.sin(angle) * jnp.cos(torsion_angle),
                                distance * jnp.sin(angle) * jnp.sin(torsion_angle)])
            vec_n = jnp.cross(vec_ab, vec_bc_norm)
            vec_n = vec_n / jnp.linalg.norm(vec_n)
            rotation_mat_m = jnp.array([vec_bc_norm, jnp.cross(vec_n, vec_bc_norm), vec_n]).T
            xyz = jax.ops.index_update(xyz, jax.ops.index[index], cell_mat_f @ (rotation_mat_m @ vec_d2 + bound_xyz))

        if type(instruction.xyz).__name__ == 'SingleTrigonalCalculated':
            bound_xyz = xyz[instruction.xyz.bound_atom_index]
            plane1_xyz = xyz[instruction.xyz.plane_atom1_index]
            plane2_xyz = xyz[instruction.xyz.plane_atom2_index]
            addition = 2 * bound_xyz - plane1_xyz - plane2_xyz
            xyz = jax.ops.index_update(xyz, jax.ops.index[index], bound_xyz + addition / jnp.linalg.norm(cell_mat_m @ addition) * instruction.xyz.distance)
        
        # constrained displacements
        if type(instruction.uij).__name__ == 'UEquivCalculated':
            uij_parent = uij[instruction.uij.atom_index, jnp.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])]
            u_cart = ucif2ucart(cell_mat_m, uij_parent[None,:, :])
            uiso = jnp.trace(u_cart) / 3
            uij = jax.ops.index_update(uij, jax.ops.index[index, :3], jnp.array([uiso, uiso, uiso]))
            uij = jax.ops.index_update(uij, jax.ops.index[index, 3], uiso * jnp.sum(cell_mat_f[:, 1] * cell_mat_f[:, 2]) / lengths_star[1] / lengths_star[2])
            uij = jax.ops.index_update(uij, jax.ops.index[index, 4], uiso * jnp.sum(cell_mat_f[:, 0] * cell_mat_f[:, 2]) / lengths_star[0] / lengths_star[2])
            uij = jax.ops.index_update(uij, jax.ops.index[index, 5], uiso * jnp.sum(cell_mat_f[:, 0] * cell_mat_f[:, 1]) / lengths_star[0] / lengths_star[1])
        elif type(instruction.uij).__name__ == 'Uiso':
            uiso = resolve_instruction(parameters, instruction.uij.uiso)
            uij = jax.ops.index_update(uij, jax.ops.index[index, :3], jnp.array([uiso, uiso, uiso]))
            uij = jax.ops.index_update(uij, jax.ops.index[index, 3], uiso * jnp.sum(cell_mat_f[:, 1] * cell_mat_f[:, 2]) / lengths_star[1] / lengths_star[2])
            uij = jax.ops.index_update(uij, jax.ops.index[index, 4], uiso * jnp.sum(cell_mat_f[:, 0] * cell_mat_f[:, 2]) / lengths_star[0] / lengths_star[2])
            uij = jax.ops.index_update(uij, jax.ops.index[index, 5], uiso * jnp.sum(cell_mat_f[:, 0] * cell_mat_f[:, 1]) / lengths_star[0] / lengths_star[1])
    return xyz, uij, cijk, dijkl, occupancies

def resolve_instruction_esd(esds, instruction):
    """Resolve fixed and refined parameter esds"""
    if type(instruction).__name__ == 'RefinedParameter':
        return_value = instruction.multiplicator * esds[instruction.par_index]
    elif type(instruction).__name__ == 'FixedParameter':
        return_value = jnp.nan # One could pick zero, but this should indicate that an error is not defined
    else:
        raise NotImplementedError('Unknown type of parameters')
    return return_value

def construct_esds(var_cov_mat, construction_instructions):
    # TODO Build analogous to the distance calculation function to get esds for all non-primitive calculations
    esds = jnp.sqrt(np.diag(var_cov_mat))
    xyz = jnp.array(
        [[resolve_instruction_esd(esds, inner_instruction) for inner_instruction in instruction.xyz]
          if type(instruction.xyz) in (tuple, list) else jnp.full(3, jnp.nan) for instruction in construction_instructions]
    )
    uij = jnp.array(
        [[resolve_instruction_esd(esds, inner_instruction) for inner_instruction in instruction.uij]
          if type(instruction.uij) in (tuple, list) else jnp.full(6, jnp.nan) for instruction in construction_instructions]
    )
    
    cijk = jnp.array(
        [[resolve_instruction_esd(esds, inner_instruction) for inner_instruction in instruction.cijk]
          if type(instruction.cijk) in (tuple, list) else jnp.full(6, jnp.nan) for instruction in construction_instructions]
    )
    
    dijkl = jnp.array(
        [[resolve_instruction_esd(esds, inner_instruction) for inner_instruction in instruction.dijkl]
          if type(instruction.dijkl) in (tuple, list) else jnp.full(6, jnp.nan) for instruction in construction_instructions]
    )
    occupancies = jnp.array([resolve_instruction_esd(esds, instruction.occupancy) for instruction in construction_instructions])
    return xyz, uij, cijk, dijkl, occupancies    

def calc_lsq_factory(cell_mat_m,
                     symm_mats_vecs,
                     index_vec_h,
                     intensities_obs,
                     weights,
                     construction_instructions,
                     fjs_core,
                     flack_parameter,
                     core_parameter,
                     extinction_parameter,
                     wavelength,
                     restraint_instr_ind=[]):
    """Generates a calc_lsq function. Doing this with a factory function allows for both flexibility but also
    speed by automatic loop and conditional unrolling for all the stuff that is constant for a given structure."""
    construct_values_j = jax.jit(construct_values, static_argnums=(1))
    cell_mat_f = jnp.linalg.inv(cell_mat_m).T
    if wavelength is not None:
        # This means a shelxl style extinction correction as that is currently the only reason to pass a wavelength
        sintheta = jnp.linalg.norm(jnp.einsum('xy, zy -> zx', cell_mat_f, index_vec_h), axis=1) / 2 * wavelength
        sintwotheta = 2 * sintheta * jnp.sqrt(1 - sintheta**2)
        extinction_factors = 0.001 * wavelength**3 / sintwotheta
    def function(parameters, fjs):
        xyz, uij, cijk, dijkl, occupancies = construct_values_j(parameters, construction_instructions, cell_mat_m)
        if fjs_core is not None:
            if core_parameter is not None:
                fjs = parameters[core_parameter] * fjs + fjs_core[None, :, :]
            else:
                fjs = fjs + fjs_core[None, :, :]

        structure_factors = calc_f(
            xyz=xyz,
            uij=uij,
            cijk=cijk,
            dijkl=dijkl,
            occupancies=occupancies,
            index_vec_h=index_vec_h,
            cell_mat_f=cell_mat_f,
            symm_mats_vecs=symm_mats_vecs,
            fjs=fjs
        )
        if extinction_parameter is None:
            intensities_calc = parameters[0] * jnp.abs(structure_factors)**2
            #restraint_addition = 0
        else:
            i_calc0 = jnp.abs(structure_factors)**2
            if wavelength is None:
                # Secondary exctinction, as shelxl needs a wavelength                
                intensities_calc = parameters[0] * i_calc0 / (1 + parameters[extinction_parameter] * i_calc0)
                #restraint_addition = 0
            else:
                intensities_calc = parameters[0] * i_calc0 / jnp.sqrt(1 + parameters[extinction_parameter] * extinction_factors * i_calc0)
                #restraint_addition = 1.0 / 0.1 * parameters[extinction_parameter]**2

        if flack_parameter is not None:
            structure_factors2 = calc_f(
                xyz=xyz,
                uij=uij,
                cijk=cijk,
                dijkl=dijkl,
                occupancies=occupancies,
                index_vec_h=-index_vec_h,
                cell_mat_f=cell_mat_f,
                symm_mats_vecs=symm_mats_vecs,
                fjs=fjs
            )
            if extinction_parameter is None:
                intensities_calc2 = parameters[0] * jnp.abs(structure_factors2)**2
                #restraint_addition = 0
            else:
                i_calc02 = jnp.abs(structure_factors2)**2
                if wavelength is None:
                    # Secondary exctinction, as shelxl needs a wavelength                
                    intensities_calc2 = parameters[0] * i_calc02 / (1 + parameters[extinction_parameter] * i_calc02)
                    #restraint_addition = 0
                else:
                    intensities_calc2 = parameters[0] * i_calc02 / jnp.sqrt(1 + parameters[extinction_parameter] * extinction_factors * i_calc02)
                    #restraint_addition = 1.0 / 0.1 * parameters[extinction_parameter]**2
            lsq = jnp.sum(weights * (intensities_obs - parameters[flack_parameter] * intensities_calc2 - (1 - parameters[flack_parameter]) * intensities_calc)**2) 
        else:
            lsq = jnp.sum(weights * (intensities_obs - intensities_calc)**2) 
        return lsq * (1 + resolve_restraints(xyz, uij, restraint_instr_ind, cell_mat_m) / (len(intensities_obs) - len(parameters)))


    return jax.jit(function)


def calc_var_cor_mat(cell_mat_m,
                     symm_mats_vecs,
                     index_vec_h,
                     construction_instructions,
                     intensities_obs,
                     weights,
                     parameters, fjs,
                     fjs_core,
                     flack_parameter,
                     core_parameter,
                     extinction_parameter,
                     wavelength):
    construct_values_j = jax.jit(construct_values, static_argnums=(1))
    cell_mat_f = jnp.linalg.inv(cell_mat_m).T
    if wavelength is not None:
        # This means a shelxl style extinction correction as that is currently the only reason to pass a wavelength
        sintheta = jnp.linalg.norm(jnp.einsum('xy, zy -> zx', cell_mat_f, index_vec_h), axis=1) / 2 * wavelength
        sintwotheta = 2 * sintheta * jnp.sqrt(1 - sintheta**2)
        extinction_factors = 0.001 * wavelength**3 / sintwotheta
    def function(parameters, fjs, index):
        xyz, uij, cijk, dijkl, occupancies = construct_values_j(parameters, construction_instructions, cell_mat_m)
        if fjs_core is not None:
            if core_parameter is not None:
                fjs = parameters[core_parameter] * fjs + fjs_core[None, :, :]
            else:
                fjs = fjs + fjs_core[None, :, :]

        structure_factors = calc_f(
            xyz=xyz,
            uij=uij,
            cijk=cijk,
            dijkl=dijkl,
            occupancies=occupancies,
            index_vec_h=index_vec_h[None, index],
            cell_mat_f=cell_mat_f,
            symm_mats_vecs=symm_mats_vecs,
            fjs=fjs[:, :, index, None]
        )
        if extinction_parameter is None:
            intensities_calc = parameters[0] * jnp.abs(structure_factors)**2
        else:
            i_calc0 = jnp.abs(structure_factors)**2
            if wavelength is None:
                # Secondary exctinction, as shelxl needs a wavelength                
                intensities_calc = parameters[0] * i_calc0 / (1 + parameters[extinction_parameter] * i_calc0)
            else:
                intensities_calc = parameters[0] * i_calc0 / jnp.sqrt(1 + parameters[extinction_parameter] * extinction_factors[index] * i_calc0)
        if flack_parameter is not None:
            structure_factors2 = calc_f(
                xyz=xyz,
                uij=uij,
                cijk=cijk,
                dijkl=dijkl,
                occupancies=occupancies,
                index_vec_h=-index_vec_h[None, index],
                cell_mat_f=cell_mat_f,
                symm_mats_vecs=symm_mats_vecs,
                fjs=fjs[:, :, index, None]
            )
            if extinction_parameter is None:
                intensities_calc2 = parameters[0] * jnp.abs(structure_factors2)**2
            else:
                i_calc02 = jnp.abs(structure_factors2)**2
                if wavelength is None:
                    # Secondary exctinction, as shelxl needs a wavelength                
                    intensities_calc2 = parameters[0] * i_calc02 / (1 + parameters[extinction_parameter] * i_calc02)
                else:
                    intensities_calc2 = parameters[0] * i_calc02 / jnp.sqrt(1 + parameters[extinction_parameter] * extinction_factors[index] * i_calc02)
            return parameters[flack_parameter] * intensities_calc2[0] - (1 - parameters[flack_parameter]) * intensities_calc[0]
        else:
            return intensities_calc[0]
    grad_func = jax.jit(jax.grad(function))

    collect = jnp.zeros((len(parameters), len(parameters)))
    for index, weight in enumerate(weights):
        val = grad_func(parameters, jnp.array(fjs), index)[:, None]
        collect += weight * (val @ val.T)

    lsq_func = calc_lsq_factory(cell_mat_m, symm_mats_vecs, index_vec_h, intensities_obs, weights, construction_instructions, fjs_core, flack_parameter, core_parameter, extinction_parameter, wavelength)
    chi_sq = lsq_func(parameters, jnp.array(fjs)) / (index_vec_h.shape[0] - len(parameters))

    return chi_sq * jnp.linalg.inv(collect)


### Internal Objects ####
AtomInstructions = namedtuple('AtomInstructions', [
    'name', # Name of atom, supposed to be unique
    'element', # Element symbol, e.g. 'Ca'
    'dispersion_real', # dispersion correction f'
    'dispersion_imag', # dispersion correction f''
    'xyz', # fractional coordinates
    'uij', # uij parameters U11, U22, U33, U23, U13, U12
    'cijk', # Gram Charlier 3rd Order C111, C222, C333, C112, C122, C113, C133, C223, C233, C123,
    'dijkl', # Gram Charlier 4th Order D1111, D2222, D3333, D1112, D1222, D1113, D1333, D2223, D2333, D1122, D1133, D2233, D1123, D1223, D1233
    'occupancy' # the occupancy
], defaults= [None, None, None, None, None, None, None, None])


RefinedParameter = namedtuple('RefinedParameter', [
    'par_index',     # index of the parameter in in the parameter array
    'multiplicator', # Multiplicator for the parameter. (default: 1.0)
    'added_value'    # Value that is added to the parameter (e.g. for occupancies)
], defaults=(None, 1, 0))


FixedParameter = namedtuple('FixedParameter', [
    'value',         # fixed value of the parameter 
    'special_position' # stems from an atom on a special position, makes a difference for output of occupancy
], defaults=[1.0, False])


UEquivCalculated = namedtuple('UEquivCalculated', [
    'atom_index',   # index of atom to set the U_equiv equal to 
    'multiplicator' # factor to multiply u_equiv with
])

UIso = namedtuple('Uiso',[
    'uiso'          # Parameter for Uiso can either be a fixed parameter or a refined Parameter
])

SingleTrigonalCalculated = namedtuple('SingleTrigonalCalculated',[
    'bound_atom_index',  # index of atom the derived atom is bound to
    'plane_atom1_index', # first bonding partner of bound_atom
    'plane_atom2_index', # second bonding partner of bound_atom
    'distance'           # interatomic distance
])

TorsionCalculated = namedtuple('TorsionCalculated', [
    'bound_atom_index',   # index of  atom the derived atom is bound_to
    'angle_atom_index',   # index of atom spanning the given angle with bound atom
    'torsion_atom_index', # index of atom giving the torsion angle
    'distance',           # interatom distance
    'angle',              # interatom angle
    'torsion_angle'       # interatom torsion angle
])

### Objects for Use by the User

ConstrainedValues = namedtuple('ConstrainedValues', [
    'variable_indexes', # 0-x (positive): variable index; -1 means 0 -> not refined
    'multiplicators', # For higher symmetries mathematical conditions can include multiplicators
    'added_value', # Values that are added
    'special_position' # stems from an atom on a special position, makes a difference for output of occupancy
], defaults=[[], [], [], False])

UEquivConstraint = namedtuple('UEquivConstraint', [
    'bound_atom', # Name of the bound atom
    'multiplicator' # Multiplicator for UEquiv Constraint (Usually nonterminal: 1.2, terminal 1.5)
])

TrigonalPositionConstraint = namedtuple('TrigonalPositionConstraint', [
    'bound_atom_name', # name of bound atom
    'plane_atom1_name', # first bonding partner of bound atom
    'plane_atom2_name', # second bonding partner of bound atom
    'distance' # interatomic distance
])

TorsionPositionConstraint = namedtuple('TorsionCalculated', [
    'bound_atom_name',   # index of  atom the derived atom is bound_to
    'angle_atom_name',   # index of atom spanning the given angle with bound atom
    'torsion_atom_name', # index of atom giving the torsion angle
    'distance',           # interatom distance
    'angle',              # interatom angle
    'torsion_angle'       # interatom torsion angle
])

def har(cell_mat_m, symm_mats_vecs, hkl, construction_instructions, parameters, f0j_source='gpaw', reload_step=1, options_dict={}, refinement_dict={}, restraints=[]):
    """
    Basic Hirshfeld atom refinement routine. Will calculate the electron density on a grid spanning the unit cell
    First will refine the scaling factor. Afterwards all other parameters defined by the parameters, 
    construction_instructions pair will be refined until 10 cycles are done or the optimizer is converged fully
    """
    print('Preparing')
    index_vec_h = jnp.array(hkl[['h', 'k', 'l']].values.copy())
    type_symbols = [atom.element for atom in construction_instructions]
    constructed_xyz, *_ = construct_values(parameters, construction_instructions, cell_mat_m)

    dispersion_real = jnp.array([atom.dispersion_real for atom in construction_instructions])
    dispersion_imag = jnp.array([atom.dispersion_imag for atom in construction_instructions])
    f_dash = dispersion_real + 1j * dispersion_imag

    if f0j_source == 'gpaw':
        from .gpaw_source import calc_f0j, calculate_f0j_core
    elif f0j_source == 'iam':
        from .iam_source import calc_f0j, calculate_f0j_core
    elif f0j_source == 'gpaw_spherical':
        from .gpaw_spherical_source import calc_f0j, calculate_f0j_core
    else:
        raise NotImplementedError('Unknown type of f0j_source')


    additional_parameters = 0
    if 'flack' in refinement_dict and refinement_dict['flack']:
        flack_parameter = additional_parameters + 1
        additional_parameters += 1
    else:
        flack_parameter = None
    if 'core' in refinement_dict:
        if f0j_source in ('iam'):
            warnings.warn('core description is not possible with this f0j source')
        if refinement_dict['core'] in ('scale', 'constant'):
            if refinement_dict['core'] == 'scale':
                core_parameter = additional_parameters + 1
                additional_parameters += 1
            else:
                core_parameter = None
            f0j_core = jnp.array(calculate_f0j_core(cell_mat_m, type_symbols, constructed_xyz, index_vec_h, symm_mats_vecs))
            f0j_core += f_dash[:, None]
        elif refinement_dict['core'] == 'fft':
            core_parameter = None
            f0j_core = None
        else:
            raise ValueError('Choose either scale, constant or fft for core description')
    else:
        core_parameter = None
        f0j_core = None
    if 'extinction' in refinement_dict:
        if refinement_dict['extinction'] == 'secondary':
            extinction_parameter = additional_parameters + 1
            additional_parameters += 1
            wavelength = None
        elif refinement_dict['extinction'] == 'none':
            extinction_parameter = None
            wavelength = None
        elif refinement_dict['extinction'] == 'shelxl':
            assert 'wavelength' in refinement_dict, 'Wavelength needs to be defined in refinement_dict for shelxl extinction'
            extinction_parameter = additional_parameters + 1
            wavelength = refinement_dict['wavelength']
            additional_parameters += 1
        else:
            raise ValueError('Choose either shelxl, secondary or none for extinction description')
    else:
        extinction_parameter = None
        wavelength = None

    if 'max_diff_recalc' in refinement_dict:
        max_distance_diff = refinement_dict['max_distance_recalc']
    else:
        max_distance_diff = 1e-6

    if 'weights' not in hkl.columns:
        hkl['weights'] = 1 / hkl['stderr']**2

    print('  calculating first atomic form factors')
    if reload_step == 0:
        restart = 'save.gpw'
    else:
        restart = None

    fjs = calc_f0j(cell_mat_m,
                   type_symbols,
                   constructed_xyz,
                   index_vec_h,
                   symm_mats_vecs,
                   gpaw_dict=options_dict,
                   save='save.gpw',
                   restart=restart,
                   explicit_core=f0j_core is not None)
    if f0j_core is None:
        fjs += f_dash[None,:,None]
    xyz_density = constructed_xyz

    print('  building least squares function')
    calc_lsq = calc_lsq_factory(cell_mat_m,
                                symm_mats_vecs,
                                jnp.array(hkl[['h', 'k', 'l']]),
                                jnp.array(hkl['intensity']),
                                jnp.array(hkl['weights']),
                                construction_instructions,
                                f0j_core,
                                flack_parameter,
                                core_parameter,
                                extinction_parameter,
                                wavelength,
                                restraints)
    print('  setting up gradients')
    grad_calc_lsq = jax.jit(jax.grad(calc_lsq))


    def minimize_scaling(x, parameters):
        parameters_new = None
        for index, value in enumerate(x):
            parameters_new = jax.ops.index_update(parameters, jax.ops.index[index], value)
        return calc_lsq(parameters_new, fjs), grad_calc_lsq(parameters_new, fjs)[:len(x)]
    print('step 0: Optimizing scaling')
    x = minimize(minimize_scaling,
                 args=(parameters.copy()),
                 x0=parameters[0],
                 jac=True,
                 options={'gtol': 1e-8 * jnp.sum(hkl["intensity"].values**2 / hkl["stderr"].values**2)})
    for index, val in enumerate(x.x):
        parameters = jax.ops.index_update(parameters, jax.ops.index[index], val)
    print(f'  wR2: {np.sqrt(x.fun / np.sum(hkl["intensity"].values**2 / hkl["stderr"].values**2)):8.6f}, nit: {x.nit}, {x.message}')

    r_opt_density = 1e10
    for refine in range(20):
        print(f'  minimizing least squares sum')
        x = minimize(calc_lsq,
                     parameters,
                     jac=grad_calc_lsq,
                     method='BFGS',
                     args=(jnp.array(fjs)),
                     options={'gtol': 1e-8 * jnp.sum(hkl["intensity"].values**2 / hkl["stderr"].values**2)})
        
        #x = jminimize(
        #    calc_lsq,
        #    x0=parameters,
        #    method='BFGS',
        #    args=(jnp.array(fjs))
        #)
        print(f'  wR2: {np.sqrt(x.fun / np.sum(hkl["intensity"].values**2 / hkl["stderr"].values**2)):8.6f}, nit: {x.nit}, {x.message}')
        shift = parameters - x.x
        parameters = jnp.array(x.x) 
        #if x.nit == 0:
        #    break
        if x.fun < r_opt_density or refine < 10:
            r_opt_density = x.fun
            #parameters_min1 = jnp.array(x.x)
        else:
            break
        #with open('save_par_model.pkl', 'wb') as fo:
        #    pickle.dump({
        #        'construction_instructions': construction_instructions,
        #        'parameters': parameters
        #    }, fo) 
        
        constructed_xyz, *_ = construct_values(parameters, construction_instructions, cell_mat_m)
        if refine >= reload_step - 1:
            restart = 'save.gpw'  
        else:
            restart = None  
        if np.max(np.linalg.norm(np.einsum('xy, zy -> zx', cell_mat_m, constructed_xyz - xyz_density), axis=-1)) > max_distance_diff:
            print(f'step {refine + 1}: calculating new structure factors')
            del(fjs)
            fjs = calc_f0j(cell_mat_m,
                           type_symbols,
                           constructed_xyz,
                           index_vec_h,
                           symm_mats_vecs,
                           restart=restart,
                           gpaw_dict=options_dict,
                           save='save.gpw',
                           explicit_core=f0j_core is not None)
            if f0j_core is None:
                fjs += f_dash[None,:,None]
            xyz_density = constructed_xyz
        else:
            print(f'step {refine + 1}: atom_positions are converged. No new structure factor calculation.')
    print('Calculation finished. calculating variance-covariance matrix.')
    var_cov_mat = calc_var_cor_mat(cell_mat_m,
                                   symm_mats_vecs,
                                   index_vec_h,
                                   construction_instructions,
                                   jnp.array(hkl['intensity']),
                                   jnp.array(hkl['weights']),
                                   parameters,
                                   fjs,
                                   f0j_core,
                                   flack_parameter,
                                   core_parameter,
                                   extinction_parameter,
                                   wavelength)
    if f0j_core is not None:
        if core_parameter is not None:
            fjs_all = parameters[core_parameter] * fjs + f0j_core[None, :, :]
        else:
            fjs_all = fjs + f0j_core[None, :, :]
    else:
        fjs_all = fjs
    shift_ov_su = shift / np.sqrt(np.diag(var_cov_mat))
    additional_information = {
        'fjs_anom': fjs_all,
        'shift_ov_su': shift_ov_su
    }
    return parameters, var_cov_mat, additional_information


def distance_with_esd(atom1_name, atom2_name, construction_instructions, parameters, var_cov_mat, cell_par, cell_std, crystal_system):
    names = [instr.name for instr in construction_instructions]
    index1 = names.index(atom1_name)
    index2 = names.index(atom2_name)

    def distance_func(parameters, cell_par):
        cell_mat_m = cell_constants_to_M(*cell_par, crystal_system)
        constructed_xyz, *_ = construct_values(parameters, construction_instructions, cell_mat_m)
        coord1 = constructed_xyz[index1]
        coord2 = constructed_xyz[index2]

        return jnp.linalg.norm(cell_mat_m @ (coord1 - coord2))
    
    distance = distance_func(parameters, cell_par)

    jac1, jac2 = jax.grad(distance_func, [0, 1])(parameters, cell_par)

    esd = jnp.sqrt(jac1[None, :] @ var_cov_mat @ jac1[None, :].T + jac2[None,:] @ jnp.diag(cell_std**2) @ jac2[None,:].T)
    return distance, esd[0, 0]


def u_iso_with_esd(atom_name, construction_instructions, parameters, var_cov_mat, cell_par, cell_std, crystal_system):
    names = [instr.name for instr in construction_instructions]
    atom_index = names.index(atom_name)
    def u_iso_func(parameters, cell_par):
        cell_mat_m = cell_constants_to_M(*cell_par, crystal_system)
        _, constructed_uij, *_ = construct_values(parameters, construction_instructions, cell_mat_m)
        cut = constructed_uij[atom_index]
        ucart = ucif2ucart(cell_mat_m, cut[None,[[0, 5, 4], [5, 1, 3], [4, 3, 2]]])
        return jnp.trace(ucart[0]) / 3
    u_iso = u_iso_func(parameters, cell_par)
    jac1, jac2 = jax.grad(u_iso_func, [0, 1])(parameters, cell_par)
    esd = jnp.sqrt(jac1[None, :] @ var_cov_mat @ jac1[None, :].T + jac2[None,:] @ jnp.diag(cell_std**2) @ jac2[None,:].T)
    return u_iso, esd[0, 0]


def angle_with_esd(atom1_name, atom2_name, atom3_name, construction_instructions, parameters, var_cov_mat, cell_par, cell_std, crystal_system):
    names = [instr.name for instr in construction_instructions]
    index1 = names.index(atom1_name)
    index2 = names.index(atom2_name)
    index3 = names.index(atom3_name)

    def angle_func(parameters, cell_par):
        cell_mat_m = cell_constants_to_M(*cell_par, crystal_system)
        constructed_xyz, *_ = construct_values(parameters, construction_instructions, cell_mat_m)
        vec1 = cell_mat_m @ (constructed_xyz[index1] - constructed_xyz[index2])
        vec2 = cell_mat_m @ (constructed_xyz[index3] - constructed_xyz[index2])

        return jnp.rad2deg(jnp.arccos((vec1 / jnp.linalg.norm(vec1)) @ (vec2 / jnp.linalg.norm(vec2))))
    
    angle = angle_func(parameters, cell_par)

    jac1, jac2 = jax.grad(angle_func, [0, 1])(parameters, cell_par)

    esd = jnp.sqrt(jac1[None, :] @ var_cov_mat @ jac1[None, :].T + jac2[None,:] @ jnp.diag(cell_std**2) @ jac2[None,:].T)
    return angle, esd[0, 0]

