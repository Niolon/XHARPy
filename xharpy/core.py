from ase.units import C
from jax._src.numpy.lax_numpy import deg2rad
from jax.config import config
from jax.core import Value
config.update('jax_enable_x64', True)

import datetime
from typing import List, Set, Dict, Tuple, Optional, Union, Any
import jax.numpy as jnp
import pandas as pd
from collections import namedtuple
import warnings
import jax
import numpy as np
import numpy.typing as npt
from copy import deepcopy
from scipy.optimize import minimize

from .restraints import resolve_restraints
from .conversion import ucif2ucart, cell_constants_to_M


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

MultiIndexParameter = namedtuple('MultiIndexParameter', [
    'par_indexes',   # tuple of indexes in the parameter array
    'multiplicators', # tuple of multiplicators for the parameter array
    'added_value' # a single added value
])

Parameter = Union[RefinedParameter, FixedParameter, MultiIndexParameter]

UEquivCalculated = namedtuple('UEquivCalculated', [
    'atom_index',   # index of atom to set the U_equiv equal to 
    'multiplicator' # factor to multiply u_equiv with
])

UIso = namedtuple('Uiso',[
    'uiso'          # Parameter for Uiso can either be a fixed parameter or a refined Parameter
], defaults=[0.1])

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
    'distance',           # interatom dpositionsistance
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

TorsionPositionConstraint = namedtuple('TorsionPositionConstraint', [
    'bound_atom_name',   # index of  atom the derived atom is bound_to
    'angle_atom_name',   # index of atom spanning the given angle with bound atom
    'torsion_atom_name', # index of atom giving the torsion angle
    'distance',           # interatom distance
    'angle',              # interatom angle
    'torsion_angle_add',  # interatom torsion angle addition. Use e.g 120° for second sp3 atom,
    'refine'              # If True torsion angle will be refined otherwise it will be fixed to torsion_angle_add
])


def expand_symm_unique(
        type_symbols: List[str],
        coordinates: npt.NDArray[np.float64],
        cell_mat_m: npt.NDArray[np.float64],
        symm_mats_vec: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        skip_symm: Dict[str, List[int]] = {},
        magmoms: Optional[npt.NDArray[np.float64]] = None
    ) -> Tuple[npt.NDArray[np.float64], List[str], npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]]]:
    """Expand the type_symbols and coordinates for one complete unit cell. Atoms on special positions
    appear only once. For disorder on a special position use skip_symm.

    Args:
        type_symbols (List[str]): Element symbols of the atoms in the asymmetric unit
        coordinates (npt.NDArray[np.float64]):  (N, 3) array of atomic coordinates
        cell_mat_m (npt.NDArray[np.float64]): Matrix with cell vectors as column vectors
        symm_mats_vec (Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]): (K, 3, 3) array of symmetry 
            matrices and (K, 3) array of translation vectors for all symmetry elements in the unit cell
        skip_symm (Dict[str, List[int]], optional): Symmetry elements with indexes given the list(s) in the 
        dictionary values with not be applied to the respective atoms with the atom names given in the key(s).
        Indexes need to be identical to the ones in symm_mats_vec. Defaults to {}.
        magmoms (Optional[npt.NDArray[np.float64]], optional): [description]. Defaults to None.

    Returns:
        npt.NDArray[np.float64]: (M, 3) array of unique atom positions within the
            unit cell
        List[str]: List of length M with element symbols for the unique atom positions within the unit cell
        npt.NDArray[np.float64]: (K, N) matrix to with indexes mapping the unique atom positions
            back to the individual symmetry elements and atom positions in the asymmetric unit
        Optional[npt.NDArray[np.float64]], optional): numpy array containing magnetic moments for
           symmetry generated atoms
    """
    symm_mats_r, symm_vecs_t = symm_mats_vec
    pos_frac0 = coordinates % 1
    un_positions = np.zeros((0, 3))
    n_atoms = 0
    type_symbols_symm = []
    inv_indexes = []
    if magmoms is not None:
        magmoms_symm = []
    else:
        magmoms_symm = None
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
        if magmoms is not None:
            magmoms_symm += [magmoms[atom_index]] * unique_indexes.shape[0]
        inv_indexes.append(inv_indexes_at + n_atoms)
        n_atoms += unique_indexes.shape[0]
    if magmoms_symm is not None:
        magmoms_symm = np.array(magmoms_symm)
    return un_positions.copy(), type_symbols_symm, np.array(inv_indexes), magmoms_symm


@jax.jit
def calc_f(
    xyz: jnp.ndarray,
    uij: jnp.ndarray,
    cijk: jnp.ndarray,
    dijkl: jnp.ndarray,
    occupancies: jnp.ndarray,
    index_vec_h: jnp.ndarray,
    cell_mat_f: jnp.ndarray,
    symm_mats_vecs: Tuple[jnp.ndarray, jnp.ndarray],
    fjs: jnp.ndarray
) -> jnp.ndarray :
    """Given a set of parameters, calculate the structure factors for all given reflections

    Args:
        xyz (jnp.ndarray): (N,3) array of fractional coordinates for the atoms in the 
            asymmetric unit
        uij (jnp.ndarray): (N, 6) array of anisotropic displacement parameters 
            (isotropic parameters have to be transformed to anitropic parameters).
            Parameters need to be in convention as used e.g. Shelxl or the cif as U.
            Order: U11, U22, U33, U23, U13, U12
        cijk (jnp.ndarray): (N, 10) array of hird-order Gram-Charlier parameters as defined
            in Inter. Tables of Cryst. B (2010): Eq 1.2.12.7. Order: C111, C222, C333,
            C112, C122, C113, C133, C223, C233, C123
        dijkl (jnp.ndarray): (N, 15) array of fourth-order Gram-Charlier parameters as defined
            in Inter. Tables of Cryst. B (2010): Eq 1.2.12.7. Order: D1111, D2222, D3333,
            D1112, D1222, D1113, D_1333, D2223, D2333, D1122, D1133, D2233, D1123, D1223,
            D1233
        occupancies (jnp.ndarray): (N) array of atomic occupancies. Atoms on special positions
            need to have an occupancy of 1/multiplicity
        index_vec_h (jnp.ndarray): (H, 3) array of Miller indicees of observed reflections
        cell_mat_f (jnp.ndarray): (3, 3) array with the reciprocal lattice vectors as
            row vectors
        symm_mats_vecs (Tuple[jnp.ndarray, jnp.ndarray]): (K, 3, 3) array of symmetry 
            matrices and (K, 3) array of translation vectors for all symmetry elements in the unit cell
        fjs (jnp.ndarray): (K, N, H) array of atomic form factors for all reflections and symmetry
            generated atoms within the unit cells. Atoms on special positions are present multiple
            times and have the atomic form factor of the full atom.

    Returns:
        jnp.ndarray: (H)-sized array with complex structure factors for each reflection
    """    
    #einsum indexes: k: n_symm, z: n_atom, h[description]: n_hkl
    lengths_star = jnp.linalg.norm(cell_mat_f, axis=0)
    symm_mats_r, symm_vecs_t = symm_mats_vecs

    vec_h_symm = jnp.einsum('zx, kxy -> kzy', index_vec_h, symm_mats_r) # vectorised version of H.T @ R
    u_mats = uij[:, jnp.array([[0, 5, 4],
                               [5, 1, 3],
                               [4, 3, 2]])]

    # Shelxl / Ucif convention
    vib_factors = jnp.exp(-2 * jnp.pi**2 * jnp.einsum('kha, khb, zab -> kzh',
                                                      vec_h_symm,
                                                      vec_h_symm,
                                                      u_mats * jnp.outer(lengths_star, lengths_star)))

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
    gram_charlier4 = (2.0 * jnp.pi**4 / 3.0) * dijkl_inner_sum
    gc_factor = 1 - gram_charlier3 + gram_charlier4

    positions_symm = jnp.einsum('kxy, zy -> kzx', symm_mats_r, xyz) + symm_vecs_t[:, None, :]
    phases = jnp.exp(2j * jnp.pi * jnp.einsum('kzx, hx -> kzh', positions_symm, index_vec_h))
    structure_factors = jnp.sum(occupancies[None, :] *  jnp.einsum('kzh, kzh, kzh, kzh -> hz', phases, vib_factors, fjs, gc_factor), axis=-1)
    return structure_factors


def resolve_instruction(parameters: jnp.ndarray, instruction: Parameter) -> jnp.ndarray:
    """Resolve a construction instruction to build the corresponding numerical value

    Args:
        parameters (jnp.ndarray): (P) array of parameters used for the refinement
        instruction (Any): Instruction of one of the known instruction type namedtuples to generate
           the numerical value from the parameters and/or the values in the instruction

    Raises:
        NotImplementedError: Type of instruction is currently not one of the known namedtuple types

    Returns:
        jnp.ndarray: a jax.numpy array of size 0 that contains the calculated value
    """    
    if type(instruction).__name__ == 'RefinedParameter':
        return_value = instruction.multiplicator * parameters[instruction.par_index] + instruction.added_value
    elif type(instruction).__name__ == 'FixedParameter':
        return_value = instruction.value
    elif type(instruction).__name__ == 'MultiIndexParameter':
        multiplicators = jnp.array(instruction.multiplicators)
        return_value = jnp.sum(multiplicators * parameters[jnp.array(instruction.par_indexes)]) + instruction.added_value
    else:
        raise NotImplementedError('This type of instruction is not implemented')
    return return_value


def constrained_values_to_instruction(
    par_index: int,
    mult: float,
    add: float,
    constraint: ConstrainedValues,
    current_index: int
) -> Parameter:
    """Convert the given constraint instruction to the internal parameter representation for the
    construction instructions.

    Args:
        par_index (int): par_index as given in the ConstrainedValues
        mult (float): multiplicator 
        add (float): added_value
        constraint (ConstrainedValues): source_constraint
        current_index (int): current index for assigning parameters.

    Returns:
        Parameter: the appropriate RefinedParameter, MultiParameter or FixedParameter object
    """
    if isinstance(par_index, (tuple, list, np.ndarray)):
        assert len(par_index) == len(mult), 'par_index and mult have different lengths'
        return MultiIndexParameter(
            par_indexes=tuple(np.array(par_index) + current_index),
            multiplicators=tuple(mult),
            added_value=float(add)
        )
    elif par_index >= 0:
        return RefinedParameter(
            par_index=int(current_index + par_index),                                                      
            multiplicator=float(mult),
            added_value=float(add)
        ) 
    else:
        return FixedParameter(value=float(add), special_position=constraint.special_position)

def is_multientry(entry: Any) -> bool:
    """Check if argument is a multientry

    Args:
        entry (Any): entry to be check

    Returns:
        bool: True if it is a multientry (list, tuple or ndarray)
    """    
    if isinstance(entry, (list, tuple)):
        return True
    elif isinstance(entry, (jnp.ndarray, np.ndarray)) and len(entry.shape) != 0:
        return True
    else: 
        return False

# construct the instructions for building the atomic parameters back from the linear parameter matrix
def create_construction_instructions(
    atom_table: pd.DataFrame,
    constraint_dict: Dict[str, Dict[str, ConstrainedValues]], 
    cell: Optional[jnp.ndarray] = None, 
    atoms_for_gc3: List[str] = [], 
    atoms_for_gc4: List[str] = [], 
    scaling0: float = 1.0, 
    exti0: float = 1e-6, 
    refinement_dict: Dict[str, Any] = {}
) -> Tuple[List[AtomInstructions], jnp.ndarray]:
    """Creates the list of atomic instructions that are used during the refinement to reconstruct the 
    atomic parameters from the parameter list.

    Args:
        atom_table (pd.DataFrame): pandas DataFrame that contains the atomic information. Columns 
            are named like their counterparts in the cif file but without the common start for
            each table (e.g. atom_site_fract_x -> fract_x). The easiest way to generate an 
            atom_table is with the cif2data function
        constraint_dict (Dict[str, Dict[str, ConstrainedValues]]): outer key is the atom label
            possible inner keys are: xyz, uij, cijk, dijkl and occ. The value of the inner 
            dict needs to be one of the possible Constraint sources 
        cell (Optional[jnp.ndarray], optional): jnp.array containing the cell parameters. Only
            necessary to calculate starting values for refined torsion angles of placed hydrogen 
            atoms. Defaults to None.
        atoms_for_gc3 (List[str], optional): List of atoms for which Gram-Charlier parameters. 
            of third order are to be refined. Defaults to [].
        atoms_for_gc4 (List[str], optional): List of atoms for which Gram-Charlier parameters. 
            of fourth order are to be refined. Defaults to [].
        scaling0 (float, optional): Starting value for the overall scaling factor. Defaults to 1.0.
        exti0 (float, optional): Starting value for the extinction correction parameter. Is only used
            if extinction is actually refined. Defaults to 1e-6.
        refinement_dict (Dict[str, Any], optional): Dictionary that contains options for the refinement.
            Defaults to {}.

    Raises:
        ValueError: Found one or more missing essential columns in atom_table
        NotImplementedError: Constraint Type is not implemented
        NotImplementedError: Uij-type is not implemented

    Returns:
        List[AtomInstructions]: list of AtomInstructions used in the refinement.
        jnp.ndarray: starting values for the parameters
    """    
    essential_columns = ['label', 'type_symbol', 'occupancy', 'fract_x', 'fract_y',
                         'fract_z', 'type_scat_dispersion_real', 'type_scat_dispersion_imag'
                         'adp_type']
    missing_columns = [c for c in essential_columns if c not in atom_table.columns]
    if len(missing_columns) > 0:
        raise ValueError(f'The following columns were missing from the atom table {", ".join(missing_columns)}')

    for gc3_atom in atoms_for_gc3:
        assert gc3_atom in list(atom_table['label']), f'Atom {gc3_atom} in Gram-Charlier 3rd order list but not in atom table'
    for gc4_atom in atoms_for_gc4:
        assert gc4_atom in list(atom_table['label']), f'Atom {gc4_atom} in Gram-Charlier 4th order list but not in atom table'
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
    if cell is None:
        cell_mat_m = None
    else:
        cell_mat_m = cell_constants_to_M(*cell)
    names = list(atom_table['label'])
    for _, atom in atom_table.iterrows():
        xyz = atom[['fract_x', 'fract_y', 'fract_z']].values.astype(np.float64)
        if atom['label'] in constraint_dict.keys() and 'xyz' in constraint_dict[atom['label']].keys():
            constraint = constraint_dict[atom['label']]['xyz']
            if type(constraint).__name__ == 'TrigonalPositionConstraint':
                bound_index = names.index(constraint.bound_atom_name)
                plane_atom1_index = names.index(constraint.plane_atom1_name)
                plane_atom2_index = names.index(constraint.plane_atom2_name)
                xyz_instructions = SingleTrigonalCalculated(bound_atom_index=bound_index,
                                                            plane_atom1_index=plane_atom1_index,
                                                            plane_atom2_index=plane_atom2_index,
                                                            distance=constraint.distance)
            elif type(constraint).__name__ == 'TorsionPositionConstraint':
                bound_index = names.index(constraint.bound_atom_name)
                angle_index = names.index(constraint.angle_atom_name)
                torsion_index = names.index(constraint.torsion_atom_name)
                index_tuple = (bound_index, angle_index, torsion_index)
                if not constraint.refine:
                    torsion_parameter_index = None
                elif index_tuple not in known_torsion_indexes:
                    assert cell_mat_m is not None, 'You need to pass a cell for the calculation of the torsion start values.'
                    atom_cart = cell_mat_m @ xyz
                    bound_xyz = atom_table.loc[bound_index,['fract_x', 'fract_y', 'fract_z']].values.astype(np.float64)
                    bound_cart = cell_mat_m @ bound_xyz
                    angle_xyz = atom_table.loc[angle_index,['fract_x', 'fract_y', 'fract_z']].values.astype(np.float64)
                    angle_cart = cell_mat_m @ angle_xyz
                    torsion_xyz = atom_table.loc[torsion_index,['fract_x', 'fract_y', 'fract_z']].values.astype(np.float64)
                    torsion_cart = cell_mat_m @ torsion_xyz
                    b1 = angle_cart- torsion_cart
                    b2 = bound_cart - angle_cart
                    b3 = atom_cart - bound_cart
                    n1 = np.cross(b1, b2)
                    n1 /= np.linalg.norm(n1)
                    n2 = np.cross(b2, b3)
                    n2 /= np.linalg.norm(n2)
                    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
                    x = np.dot(n1, n2)
                    y = np.dot(m1, n2)
                    torsion0 = np.arctan2(y, x) - np.deg2rad(constraint.torsion_angle_add)
                    parameters = jax.ops.index_update(parameters, jax.ops.index[current_index], torsion0)
                    known_torsion_indexes[index_tuple] = current_index
                    torsion_parameter_index = current_index
                    current_index += 1
                else:
                    torsion_parameter_index = known_torsion_indexes[index_tuple]
                
                if constraint.refine:
                    torsion_parameter = RefinedParameter(
                        par_index=torsion_parameter_index,
                        multiplicator=1.0,
                        added_value=np.deg2rad(constraint.torsion_angle_add)
                    )
                else:
                    torsion_parameter = FixedParameter(
                        value=np.deg2rad(constraint.torsion_angle_add)
                    )
                xyz_instructions = TorsionCalculated(
                    bound_atom_index=bound_index,
                    angle_atom_index=angle_index,
                    torsion_atom_index=torsion_index,
                    distance=FixedParameter(value=float(constraint.distance)),
                    angle=FixedParameter(value=float(constraint.angle)),
                    torsion_angle=torsion_parameter
                )
            elif type(constraint).__name__ == 'ConstrainedValues':
                instr_zip = zip(constraint.variable_indexes, constraint.multiplicators, constraint.added_value)
                xyz_instructions = tuple(constrained_values_to_instruction(par_index, mult, add, constraint, current_index) for par_index, mult, add in instr_zip)
                # we need this construction to unpack lists in indexes for the MultiIndexParameters
                n_pars = max(max(entry) if is_multientry(entry) else entry for entry in constraint.variable_indexes) + 1
                # MultiIndexParameter can never be unique so we can throw it out
                u_indexes = [-1 if is_multientry(entry) else entry for entry in constraint.variable_indexes]
                parameters = jax.ops.index_update(
                    parameters,
                    jax.ops.index[current_index:current_index + n_pars],
                    [xyz[jnp.array(varindex)] for index, varindex in zip(*np.unique(u_indexes, return_index=True)) if index >=0]
                )
                current_index += n_pars
            else:
                raise(NotImplementedError(f'Unknown type of xyz constraint for atom {atom["label"]}'))
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
                    adp_instructions = tuple(constrained_values_to_instruction(par_index, mult, add, constraint, current_index) for par_index, mult, add in instr_zip)
                    # we need this construction to unpack lists in indexes for the MultiIndexParameters
                    n_pars = max(max(entry) if is_multientry(entry) else entry for entry in constraint.variable_indexes) + 1

                    # MultiIndexParameter can never be unique so we can throw it out
                    u_indexes = [-1 if is_multientry(entry) else entry for entry in constraint.variable_indexes]

                    parameters = jax.ops.index_update(
                        parameters,
                        jax.ops.index[current_index:current_index + n_pars],
                        [adp[jnp.array(varindex)] for index, varindex in zip(*np.unique(u_indexes, return_index=True)) if index >=0]
                    )
                    current_index += n_pars
                elif type(constraint).__name__ == 'UEquivConstraint':
                    bound_index = names.index(constraint.bound_atom)
                    adp_instructions = UEquivCalculated(atom_index=bound_index, multiplicator=constraint.multiplicator)
                else:
                    raise NotImplementedError('Unknown Uij Constraint')
            else:
                parameters = jax.ops.index_update(parameters, jax.ops.index[current_index:current_index + 6], list(adp))
                adp_instructions = tuple(RefinedParameter(par_index=int(array_index), multiplicator=1.0) for array_index in range(current_index, current_index + 6))
                current_index += 6
        elif atom['adp_type'] == 'Uiso':
            adp_instructions = UIso(uiso=RefinedParameter(par_index=int(current_index), multiplicator=1.0))
            parameters = jax.ops.index_update(parameters, jax.ops.index[current_index], float(atom['U_iso_or_equiv']))
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
            cijk_instructions = tuple(constrained_values_to_instruction(par_index, mult, add, constraint, current_index) for par_index, mult, add in instr_zip)
            # we need this construction to unpack lists in indexes for the MultiIndexParameters
            n_pars = max(max(entry) if is_multientry(entry) else entry for entry in constraint.variable_indexes) + 1

            # MultiIndexParameter can never be unique so we can throw it out
            u_indexes = [-1 if is_multientry(entry) else entry for entry in constraint.variable_indexes]

            parameters = jax.ops.index_update(
                parameters,
                jax.ops.index[current_index:current_index + n_pars],
                [cijk[jnp.array(varindex)] for index, varindex in zip(*np.unique(u_indexes, return_index=True)) if index >=0]
            )

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
            dijkl_instructions = tuple(constrained_values_to_instruction(par_index, mult*1e-3, add, constraint, current_index) for par_index, mult, add in instr_zip)
            # we need this construction to unpack lists in indexes for the MultiIndexParameters
            n_pars = max(max(entry) if is_multientry(entry) else entry for entry in constraint.variable_indexes) + 1
            # MultiIndexParameter can never be unique so we can throw it out
            u_indexes = [-1 if is_multientry(entry) else entry for entry in constraint.variable_indexes]
            parameters = jax.ops.index_update(
                parameters,
                jax.ops.index[current_index:current_index + n_pars],
                [dijkl[jnp.array(varindex)]*1e3 for index, varindex in zip(*np.unique(u_indexes, return_index=True)) if index >=0]
            )
            current_index += n_pars
        elif atom['label'] in atoms_for_gc4:
            parameters = jax.ops.index_update(parameters, jax.ops.index[current_index:current_index + 15], list(dijkl*1e3))
            dijkl_instructions = tuple(RefinedParameter(par_index=int(array_index), multiplicator=1e-3) for array_index in range(current_index, current_index + 15))
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



def construct_values(
    parameters: jnp.ndarray,
    construction_instructions: List[AtomInstructions],
    cell_mat_m: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reconstruct xyz, adp-parameters and occupancies from the given construction instructions. Allows for the
    Flexible usage of combinations of fixed parameters and parameters that are refined, as well as constraints

    Args:
        parameters (jnp.ndarray): parameters used during the refinement.
        construction_instructions (List[AtomInstructions]): List of instructions for reconstructing the atomic
            parameters
        cell_mat_m (jnp.ndarray): (3, 3) array with the cell vectors as row vectors, used for Uiso calculation.

    Returns:
        jnp.ndarray: (N,3) array of fractional coordinates for the atoms in the 
            asymmetric unit
        jnp.ndarray: (N, 6) array of anisotropic displacement parameters 
            (isotropic parameters are transformed to anitropic parameters).
            Parameters need to be in convention as used e.g. Shelxl or the cif as U.
            Order: U11, U22, U33, U23, U13, U12
        jnp.ndarray: (N, 10) array of hird-order Gram-Charlier parameters as defined
            in Inter. Tables of Cryst. B (2010): Eq 1.2.12.7. Order: C111, C222, C333,
            C112, C122, C113, C133, C223, C233, C123
        np.ndarray: (N, 15) array of fourth-order Gram-Charlier parameters as defined
            in Inter. Tables of Cryst. B (2010): Eq 1.2.12.7. Order: D1111, D2222, D3333,
            D1112, D1222, D1113, D_1333, D2223, D2333, D1122, D1133, D2233, D1123, D1223,
            D1233
        jnp.ndarray: (N) array of atomic occupancies. Atoms on special positions
            have an occupancy of 1/multiplicity
    """
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
            angle = resolve_instruction(parameters, instruction.xyz.angle)
            torsion_angle = resolve_instruction(parameters, instruction.xyz.torsion_angle)
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

def resolve_instruction_esd(
    var_cov_mat: jnp.ndarray,
    instruction: Parameter
) -> float:
    """Calculates the estimated standard deviation (esd) for a value calculated from a parameter

    Args:
        var_cov_mat (jnp.ndarray): (P, P) array containing the variances and covariances, 
            where P is the number of refined parameters.
        instruction (Parameter): one parameter instruction, as used for generating atomic
            parameters (and in this case esds) from the refined parameters

    Raises:
        NotImplementedError: Unknown type of Parameter used

    Returns:
        float: esd as float or np.nan if parameter has no esd
    """
    if type(instruction).__name__ == 'RefinedParameter':
        return_value = instruction.multiplicator * np.sqrt(var_cov_mat[instruction.par_index, instruction.par_index])
    elif type(instruction).__name__ == 'FixedParameter':
        return_value = jnp.nan # One could pick zero, but this should indicate that an error is not defined
    elif type(instruction).__name__ == 'MultiIndexParameter':
        jac = jnp.array(instruction.multiplicators)
        indexes = jnp.array(instruction.par_indexes)
        return_value = jnp.sqrt(jnp.sqrt(jac[None, :] @ var_cov_mat[indexes][: , indexes] @ jac[None, :].T))[0,0]
    else:
        raise NotImplementedError('Unknown type of parameters')
    return return_value

def construct_esds(var_cov_mat, construction_instructions):
    # TODO Build analogous to the distance calculation function to get esds for all non-primitive calculations
    xyz = jnp.array(
        [[resolve_instruction_esd(var_cov_mat, inner_instruction) for inner_instruction in instruction.xyz]
          if type(instruction.xyz) in (tuple, list, np.ndarray, jnp.ndarray) else jnp.full(3, jnp.nan) 
          for instruction in construction_instructions]
    )
    uij = jnp.array(
        [[resolve_instruction_esd(var_cov_mat, inner_instruction) for inner_instruction in instruction.uij]
          if type(instruction.uij) in (tuple, list, np.ndarray, jnp.ndarray) else jnp.full(6, jnp.nan) 
          for instruction in construction_instructions]
    )
    
    cijk = jnp.array(
        [[resolve_instruction_esd(var_cov_mat, inner_instruction) for inner_instruction in instruction.cijk]
          if type(instruction.cijk) in (tuple, list, np.ndarray, jnp.ndarray) else jnp.full(6, jnp.nan) 
          for instruction in construction_instructions]
    )
    
    dijkl = jnp.array(
        [[resolve_instruction_esd(var_cov_mat, inner_instruction) for inner_instruction in instruction.dijkl]
          if type(instruction.dijkl) in (tuple, list, np.ndarray, jnp.ndarray) else jnp.full(6, jnp.nan) 
          for instruction in construction_instructions]
    )
    occupancies = jnp.array([resolve_instruction_esd(var_cov_mat, instruction.occupancy) for instruction in construction_instructions])
    return xyz, uij, cijk, dijkl, occupancies    

def calc_lsq_factory(cell_mat_m,
                     symm_mats_vecs,
                     index_vec_h,
                     intensities_obs,
                     weights,
                     construction_instructions,
                     fjs_core,
                     refinement_dict,
                     #flack_parameter,
                     #core_parameter,
                     #extinction_parameter,
                     #wavelength,
                     restraint_instr_ind=[]):
    """Generates a calc_lsq function. Doing this with a factory function allows for both flexibility but also
    speed by automatic loop and conditional unrolling for all the stuff that is constant for a given structure."""
    cell_mat_f = jnp.linalg.inv(cell_mat_m).T
    additional_parameters = 0
    if refinement_dict.get('flack', False):
        refine_flack = True
        flack_parameter = additional_parameters + 1
        additional_parameters += 1
    core = refinement_dict.get('core', 'constant')
    if  core == 'scale':
        core_parameter = additional_parameters + 1
        additional_parameters += 1
    extinction = refinement_dict.get('extinction', 'none')
    if  extinction == 'shelxl':
        assert 'wavelength' in refinement_dict, 'Wavelength needs to be defined in refinement_dict for shelxl extinction'
        extinction_parameter = additional_parameters + 1
        wavelength = refinement_dict['wavelength']
        additional_parameters += 1
        sintheta = jnp.linalg.norm(jnp.einsum('xy, zy -> zx', cell_mat_f, index_vec_h), axis=1) / 2 * wavelength
        sintwotheta = 2 * sintheta * jnp.sqrt(1 - sintheta**2)
        extinction_factors = 0.001 * wavelength**3 / sintwotheta
    elif extinction == 'secondary':
        extinction_parameter = additional_parameters + 1
        additional_parameters += 1
    elif extinction == 'none':
        pass
    else:
        raise NotImplementedError('Extinction method not implemented in lsq_factory')
    
    construct_values_j = jax.jit(construct_values, static_argnums=(1))

    def function(parameters, fjs):
        xyz, uij, cijk, dijkl, occupancies = construct_values_j(parameters, construction_instructions, cell_mat_m)
        if core == 'scale':
            fjs = parameters[core_parameter] * fjs + fjs_core[None, :, :]
        elif core == 'constant':
            fjs = fjs + fjs_core[None, :, :]
        elif core == 'combine':
            pass
        else:
            raise NotImplementedError('core description not implemented in lsq_factory')

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
        if extinction == 'none':
            intensities_calc = parameters[0] * jnp.abs(structure_factors)**2
            #restraint_addition = 0
        elif extinction == 'shelxl':
            i_calc0 = jnp.abs(structure_factors)**2
            intensities_calc = parameters[0] * i_calc0 / jnp.sqrt(1 + parameters[extinction_parameter] * extinction_factors * i_calc0)
        elif extinction == 'secondary':
            i_calc0 = jnp.abs(structure_factors)**2             
            intensities_calc = parameters[0] * i_calc0 / (1 + parameters[extinction_parameter] * i_calc0)
            #restraint_addition = 0
  
        if refine_flack:
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
            if extinction == 'none':
                intensities_calc2 = parameters[0] * jnp.abs(structure_factors2)**2
                #restraint_addition = 0
            elif extinction == 'shelxl':
                i_calc02 = jnp.abs(structure_factors2)**2
                intensities_calc2 = parameters[0] * i_calc02 / jnp.sqrt(1 + parameters[extinction_parameter] * extinction_factors * i_calc02)
            elif extinction == 'secondary':
                i_calc02 = jnp.abs(structure_factors2)**2             
                intensities_calc2 = parameters[0] * i_calc02 / (1 + parameters[extinction_parameter] * i_calc02)
                #restraint_addition = 0
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
                     parameters,
                     fjs,
                     fjs_core,
                     refinement_dict):
                     #flack_parameter,
                     #core_parameter,
                     #extinction_parameter,
                     #waveleng    
    cell_mat_f = jnp.linalg.inv(cell_mat_m).T
    additional_parameters = 0
    if refinement_dict.get('flack', False):
        refine_flack = True
        flack_parameter = additional_parameters + 1
        additional_parameters += 1
    core = refinement_dict.get('core', 'constant')
    if  core == 'scale':
        core_parameter = additional_parameters + 1
        additional_parameters += 1
    extinction = refinement_dict.get('extinction', 'none')
    if  extinction == 'shelxl':
        assert 'wavelength' in refinement_dict, 'Wavelength needs to be defined in refinement_dict for shelxl extinction'
        extinction_parameter = additional_parameters + 1
        wavelength = refinement_dict['wavelength']
        additional_parameters += 1
        sintheta = jnp.linalg.norm(jnp.einsum('xy, zy -> zx', cell_mat_f, index_vec_h), axis=1) / 2 * wavelength
        sintwotheta = 2 * sintheta * jnp.sqrt(1 - sintheta**2)
        extinction_factors = 0.001 * wavelength**3 / sintwotheta
    elif extinction == 'secondary':
        extinction_parameter = additional_parameters + 1
        additional_parameters += 1
    elif extinction == 'none':
        pass
    else:
        raise NotImplementedError('Extinction method not implemented in var_cov_mat_func')
    construct_values_j = jax.jit(construct_values, static_argnums=(1))

    def function(parameters, fjs, index):
        xyz, uij, cijk, dijkl, occupancies = construct_values_j(parameters, construction_instructions, cell_mat_m)
        if core == 'scale':
            fjs = parameters[core_parameter] * fjs + fjs_core[None, :, :]
        elif core == 'constant':
            fjs = fjs + fjs_core[None, :, :]
        elif core == 'combine':
            pass
        else:
            raise NotImplementedError('core description not implemented in lsq_factory')

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
        if extinction == 'none':
            intensities_calc = parameters[0] * jnp.abs(structure_factors)**2
            #restraint_addition = 0
        elif extinction == 'shelxl':
            i_calc0 = jnp.abs(structure_factors)**2
            intensities_calc = parameters[0] * i_calc0 / jnp.sqrt(1 + parameters[extinction_parameter] * extinction_factors[index] * i_calc0)
        elif extinction == 'secondary':
            i_calc0 = jnp.abs(structure_factors)**2             
            intensities_calc = parameters[0] * i_calc0 / (1 + parameters[extinction_parameter] * i_calc0)
            #restraint_addition = 0
            
        if refine_flack:
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
            if extinction == 'none':
                intensities_calc2 = parameters[0] * jnp.abs(structure_factors2)**2
                #restraint_addition = 0
            elif extinction == 'shelxl':
                i_calc02 = jnp.abs(structure_factors2)**2
                intensities_calc2 = parameters[0] * i_calc02 / jnp.sqrt(1 + parameters[extinction_parameter] * extinction_factors[index] * i_calc02)
            elif extinction == 'secondary':
                i_calc02 = jnp.abs(structure_factors2)**2             
                intensities_calc2 = parameters[0] * i_calc02 / (1 + parameters[extinction_parameter] * i_calc02)
                #restraint_addition = 0
            return parameters[flack_parameter] * intensities_calc2[0] - (1 - parameters[flack_parameter]) * intensities_calc[0]
        else:
            return intensities_calc[0]
    grad_func = jax.jit(jax.grad(function))

    collect = jnp.zeros((len(parameters), len(parameters)))

    # TODO: Figure out a way to make this more efficient
    for index, weight in enumerate(weights):
        val = grad_func(parameters, jnp.array(fjs), index)[:, None]
        collect += weight * (val @ val.T)

    lsq_func = calc_lsq_factory(cell_mat_m, symm_mats_vecs, index_vec_h, intensities_obs, weights, construction_instructions, fjs_core, flack_parameter, core_parameter, extinction_parameter, wavelength)
    chi_sq = lsq_func(parameters, jnp.array(fjs)) / (index_vec_h.shape[0] - len(parameters))

    return chi_sq * jnp.linalg.inv(collect)

def har(
    cell,
    symm_mats_vecs, 
    hkl, 
    construction_instructions, 
    parameters, 
    f0j_source='gpaw', 
    reload_step=1, 
    options_dict={}, 
    refinement_dict={}, 
    restraints=[]
):
    """
    Basic Hirshfeld atom refinement routine. Will calculate the electron density on a grid spanning the unit cell
    First will refine the scaling factor. Afterwards all other parameters defined by the parameters, 
    construction_instructions pair will be refined until 10 cycles are done or the optimizer is converged fully
    """
    start = datetime.datetime.now()
    print('Started refinement at ', start)
    cell_mat_m = cell_constants_to_M(*cell)
    options_dict = deepcopy(options_dict)
    print('Preparing')
    index_vec_h = jnp.array(hkl[['h', 'k', 'l']].values.copy())
    type_symbols = [atom.element for atom in construction_instructions]
    constructed_xyz, constructed_uij, *_ = construct_values(parameters, construction_instructions, cell_mat_m)

    dispersion_real = jnp.array([atom.dispersion_real for atom in construction_instructions])
    dispersion_imag = jnp.array([atom.dispersion_imag for atom in construction_instructions])
    f_dash = dispersion_real + 1j * dispersion_imag

    if f0j_source == 'gpaw':
        from .gpaw_source import calc_f0j, calculate_f0j_core
    elif f0j_source == 'iam':
        from .iam_source import calc_f0j, calculate_f0j_core
    elif f0j_source == 'gpaw_spherical':
        from .gpaw_spherical_source import calc_f0j, calculate_f0j_core
    elif f0j_source == 'gpaw_lcorr':
        from .gpaw_source_lcorr import calc_f0j, calculate_f0j_core
    elif f0j_source == 'gpaw_mbis':
        from .gpaw_mbis_source import calc_f0j, calculate_f0j_core
    elif f0j_source == 'qe':
        from .qe_source import calc_f0j, calculate_f0j_core
    elif f0j_source == 'gpaw_mpi':
        from .gpaw_mpi_source import calc_f0j, calculate_f0j_core
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
            if f0j_source == 'qe':
                f0j_core, options_dict = calculate_f0j_core(cell_mat_m, type_symbols, index_vec_h, options_dict)
                f0j_core = jnp.array(f0j_core)
            else:
                f0j_core = jnp.array(calculate_f0j_core(cell_mat_m, type_symbols, constructed_xyz, index_vec_h, symm_mats_vecs))
            f0j_core += f_dash[:, None]
        elif refinement_dict['core'] == 'combine':
            core_parameter = None
            f0j_core = None
        else:
            raise ValueError('Choose either scale, constant or combine for core description')
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
        hkl['weights'] = 1 / hkl['esd_int']**2

    print('  calculating first atomic form factors')
    if reload_step == 0:
        restart = 'save.gpw'
    else:
        restart = None
    if f0j_source == 'gpaw_lcorr':
        fjs = calc_f0j(cell_mat_m,
                       type_symbols,
                       constructed_xyz,
                       constructed_uij,
                       index_vec_h,
                       symm_mats_vecs,
                       options_dict=options_dict,
                       save='save.gpw',
                       restart=restart,
                       explicit_core=f0j_core is not None)
    else:
        fjs = calc_f0j(cell_mat_m,
                       type_symbols,
                       constructed_xyz,
                       index_vec_h,
                       symm_mats_vecs,
                       options_dict=options_dict,
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
                 options={'gtol': 1e-8 * jnp.sum(hkl["intensity"].values**2 / hkl["esd_int"].values**2)})
    for index, val in enumerate(x.x):
        parameters = jax.ops.index_update(parameters, jax.ops.index[index], val)
    print(f'  wR2: {np.sqrt(x.fun / np.sum(hkl["intensity"].values**2 / hkl["esd_int"].values**2)):8.6f}, nit: {x.nit}, {x.message}')

    r_opt_density = 1e10
    for refine in range(20):
        print(f'  minimizing least squares sum')
        x = minimize(calc_lsq,
                     parameters,
                     jac=grad_calc_lsq,
                     method='BFGS',
                     args=(jnp.array(fjs)),
                     options={'gtol': 1e-8 * jnp.sum(hkl["intensity"].values**2 / hkl["esd_int"].values**2)})
        
        #x = jminimize(
        #    calc_lsq,
        #    x0=parameters,
        #    method='BFGS',
        #    args=(jnp.array(fjs))
        #)
        print(f'  wR2: {np.sqrt(x.fun / np.sum(hkl["intensity"].values**2 / hkl["esd_int"].values**2)):8.6f}, nit: {x.nit}, {x.message}')
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
        
        constructed_xyz, constructed_uij, *_ = construct_values(parameters, construction_instructions, cell_mat_m)
        if refine >= reload_step - 1:
            restart = 'save.gpw'  
        else:
            restart = None  
        if np.max(np.linalg.norm(np.einsum('xy, zy -> zx', cell_mat_m, constructed_xyz - xyz_density), axis=-1)) > max_distance_diff:
            print(f'step {refine + 1}: calculating new structure factors')
            del(fjs)
            if f0j_source == 'gpaw_lcorr':
                fjs = calc_f0j(cell_mat_m,
                            type_symbols,
                            constructed_xyz,
                            constructed_uij,
                            index_vec_h,
                            symm_mats_vecs,
                            options_dict=options_dict,
                            save='save.gpw',
                            restart=restart,
                            explicit_core=f0j_core is not None)
            else:
                fjs = calc_f0j(cell_mat_m,
                            type_symbols,
                            constructed_xyz,
                            index_vec_h,
                            symm_mats_vecs,
                            options_dict=options_dict,
                            save='save.gpw',
                            restart=restart,
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
    end = datetime.datetime.now()
    print('Ended refinement at ', end)

    additional_information = {
        'fjs_anom': fjs_all,
        'shift_ov_su': shift_ov_su,
        'start': start,
        'end': end
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

