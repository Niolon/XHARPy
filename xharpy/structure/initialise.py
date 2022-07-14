from collections import namedtuple
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List

from ..conversion import cell_constants_to_M
from ..defaults import get_parameter_index
from .common import jnp
from .common import (
    AtomInstructions, FixedParameter, RefinedParameter, MultiIndexParameter,
    Parameter, Array
)

from .positions import SingleTrigonalCalculated, TetrahedralCalculated, TorsionCalculated
from .displacements import IsoTFactor, UEquivTFactor, AnisoTFactor

CommonOccupancyParameter = namedtuple('CommonOccupancyParameter', [
    'label'        , # any identifying label that is immutable
    'multiplicator', # Multiplicator for the parameter value
    'added_value',   # Value that is added to the parameter
    'special_position'
])

ConstrainedValues = namedtuple('ConstrainedValues', [
    'variable_indexes', # 0-x (positive): variable index; -1 means 0 
                        # -> not refined
    'multiplicators',   # For higher symmetries mathematical conditions can
                        # include multiplicators
    'added_value',      # Values that are added
    'special_position'  # stems from an atom on a special position,
                        # makes a difference for output of occupancy
], defaults=[[], [], [], False])

UEquivConstraint = namedtuple('UEquivConstraint', [
    'bound_atom_name', # Name of the bound atom
    'multiplicator'    # Multiplicator for UEquiv Constraint
                       # (Usually nonterminal: 1.2, terminal 1.5)
])

TrigonalPositionConstraint = namedtuple('TrigonalPositionConstraint', [
    'bound_atom_name',  # name of bound atom
    'plane_atom1_name', # first bonding partner of bound atom
    'plane_atom2_name', # second bonding partner of bound atom
    'distance'          # interatomic distance
])

TetrahedralPositionConstraint = namedtuple('TetrahedralPositionConstraint', [
    'bound_atom_name',        # name of bound atom 
    'tetrahedron_atom1_name', # name of first atom forming the tetrahedron
    'tetrahedron_atom2_name', # name of second atom forming the tetrahedron
    'tetrahedron_atom3_name', # name of third atom forming the tetrahedron
    'distance'                # interatomic distance
])

TorsionPositionConstraint = namedtuple('TorsionPositionConstraint', [
    'bound_atom_name',   # index of  atom the derived atom is bound_to
    'angle_atom_name',   # index of atom spanning the given angle with bound atom
    'torsion_atom_name', # index of atom giving the torsion angle
    'distance',          # interatom distance
    'angle',             # interatom angle
    'torsion_angle_add', # interatom torsion angle addition.
                         # Use e.g 120° for second sp3 atom,
    'refine'             # If True torsion angle will be refined otherwise it 
                         #will be fixed to torsion_angle_add
])


def constrained_values_to_instruction(
    par_index: int,
    mult: float,
    add: float,
    constraint: ConstrainedValues,
    current_index: int
) -> Parameter:
    """Convert the given constraint instruction to the internal parameter 
    representation for the construction instructions.

    Parameters
    ----------
    par_index : int
        par_index as given in the ConstrainedValues
    mult : float
        multiplicator given in ConstrainedValues
    add : float
        added_value given in ConstrainedValues
    constraint : ConstrainedValues
        Constraint that is the source of the parameter constraint
    current_index : int
        current index for assigning parameters.

    Returns
    -------
    instruction: Parameter
        the appropriate RefinedParameter, MultiParameter or FixedParameter
        object
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
    """Check if argument is a multientry"""    
    if isinstance(entry, (list, tuple)):
        return True
    elif isinstance(entry, (jnp.ndarray, np.ndarray)) and len(entry.shape) != 0:
        return True
    else: 
        return False



def create_construction_instructions(
    atom_table: pd.DataFrame,
    refinement_dict: Dict[str, Any],
    constraint_dict: Dict[str, Dict[str, ConstrainedValues]], 
    cell: Optional[jnp.ndarray] = None, 
    atoms_for_gc3: List[str] = [], 
    atoms_for_gc4: List[str] = [], 
    scaling0: float = 1.0, 
    exti0: float = 1e-6, 
) -> Tuple[List[AtomInstructions], jnp.ndarray]:
    """Creates the list of atomic instructions that are used during the
    refinement to reconstruct the atomic parameters from the parameter list.

    Parameters
    ----------
    atom_table : pd.DataFrame
        pandas DataFrame that contains the atomic information. Columns are named
        like their counterparts in the cif file but without the common start for
        each table (e.g. atom_site_fract_x -> fract_x). The easiest way to
        generate an atom_table is with the cif2data function
    refinement_dict : Dict[str, Any]
        Dictionary that contains options for the refinement
    constraint_dict : Dict[str, Dict[str, ConstrainedValues]]
        outer key is the atom label. possible inner keys are: xyz, uij, cijk,
        dijkl and occ. The value of the inner dict needs to be one of the
        possible Constraint sources 
    cell : Optional[jnp.ndarray], optional
        jnp.array containing the cell parameters. Necessary if constraints 
        involving distances or U(equiv) are used, by default None
    atoms_for_gc3 : List[str], optional
        List of atoms for which Gram-Charlier parametersof third order are to be
        refined, by default []
    atoms_for_gc4 : List[str], optional
        List of atoms for which Gram-Charlier parameters of fourth order are to
        be refined, by default []
    scaling0 : float, optional
        Starting value for the overall scaling factor, by default 1.0
    exti0 : float, optional
        Starting value for the extinction correction parameter. Is only used
        if extinction is actually refined, by default 1e-6

    Returns
    -------
    construction_instructions: List[AtomInstructions]
        list of AtomInstructions used in the refinement.
    parameters0: jnp.ndarray: 
        starting values for the parameters

    Raises
    ------
    ValueError
        Found one or more missing essential columns in atom_table
    NotImplementedError
        Constraint Type is not implemented
    NotImplementedError
        Uij-type is not implemented
    """
  
    essential_columns = ['label', 'type_symbol', 'occupancy', 'fract_x', 'fract_y',
                         'fract_z', 'type_scat_dispersion_real', 'type_scat_dispersion_imag',
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
    parameters = parameters.at[0].set(scaling0)
    flack_index = get_parameter_index('flack', refinement_dict)
    if flack_index is not None:
        parameters = parameters.at[flack_index].set(0.0)
        current_index += 1
    core_index = get_parameter_index('core', refinement_dict)
    if core_index is not None:
        parameters = parameters.at[core_index].set(1.0)
        current_index += 1
    extinction_index = get_parameter_index('extinction', refinement_dict)
    if extinction_index is not None:
        parameters = parameters.at[extinction_index].set(exti0)
        current_index += 1
    known_torsion_indexes = {}
    if cell is None:
        cell_mat_m = None
    else:
        cell_mat_m = cell_constants_to_M(*cell)
    names = list(atom_table['label'])
    common_occupancy_indexes = {}

    construction_instructions = [None] * len(atom_table)
    for iteration in range(10):
        # we need to go through the list multiple times sa there could be interdependencies
        for instruction_index, (_, atom) in enumerate(atom_table.iterrows()):
            if construction_instructions[instruction_index] is not None:
                continue
            xyz = atom[['fract_x', 'fract_y', 'fract_z']].values.astype(np.float64)
            if atom['label'] in constraint_dict.keys() and 'xyz' in constraint_dict[atom['label']].keys():
                constraint = constraint_dict[atom['label']]['xyz']
                # TODO Make the Constraints used by the user dataclasses as well
                if type(constraint).__name__ == 'TrigonalPositionConstraint':
                    bound_index = names.index(constraint.bound_atom_name)
                    plane_atom1_index = names.index(constraint.plane_atom1_name)
                    plane_atom2_index = names.index(constraint.plane_atom2_name)
                    check_indexes = (bound_index, plane_atom1_index, plane_atom2_index)
                    if any(construction_instructions[i] is None for i in check_indexes):
                        continue
                    xyz_instructions = SingleTrigonalCalculated(
                        bound_atom_index=bound_index,
                        plane_atom1_index=plane_atom1_index,
                        plane_atom2_index=plane_atom2_index,
                        distance_par=FixedParameter(value=float(constraint.distance))
                    )
                elif type(constraint).__name__ == 'TetrahedralPositionConstraint':
                    bound_index = names.index(constraint.bound_atom_name)
                    tet1_index = names.index(constraint.tetrahedron_atom1_name)
                    tet2_index = names.index(constraint.tetrahedron_atom2_name)
                    tet3_index = names.index(constraint.tetrahedron_atom3_name)
                    check_indexes = (bound_index, tet1_index, tet2_index, tet3_index)
                    if any(construction_instructions[i] is None for i in check_indexes):
                        continue
                    xyz_instructions = TetrahedralCalculated(
                        bound_atom_index=bound_index,
                        tetrahedron_atom1_index=tet1_index,
                        tetrahedron_atom2_index=tet2_index,
                        tetrahedron_atom3_index=tet3_index,
                        distance_par=FixedParameter(value=float(constraint.distance))
                    )
                elif type(constraint).__name__ == 'TorsionPositionConstraint':
                    bound_index = names.index(constraint.bound_atom_name)
                    angle_index = names.index(constraint.angle_atom_name)
                    torsion_index = names.index(constraint.torsion_atom_name)
                    index_tuple = (bound_index, angle_index, torsion_index)
                    if any(construction_instructions[i] is None for i in index_tuple):
                        continue
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
                        parameters = parameters.at[current_index].set(torsion0)
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
                        distance_par=FixedParameter(value=float(constraint.distance)),
                        angle_par=FixedParameter(value=np.deg2rad(float(constraint.angle))),
                        torsion_angle_par=torsion_parameter
                    )
                elif type(constraint).__name__ == 'ConstrainedValues':
                    instr_zip = zip(constraint.variable_indexes, constraint.multiplicators, constraint.added_value)
                    xyz_instructions = Array(
                        parameter_tuple=tuple(constrained_values_to_instruction(par_index, mult, add, constraint, current_index) for par_index, mult, add in instr_zip)
                    )
                    # we need this construction to unpack lists in indexes for the MultiIndexParameters
                    n_pars = max(max(entry) if is_multientry(entry) else entry for entry in constraint.variable_indexes) + 1
                    # MultiIndexParameter can never be unique so we can throw it out
                    u_indexes = [-1 if is_multientry(entry) else entry for entry in constraint.variable_indexes]
                    parameters = parameters.at[current_index:current_index + n_pars].set(
                        [xyz[jnp.array(varindex)] for index, varindex in zip(*np.unique(u_indexes, return_index=True)) if index >=0]
                    )
                    current_index += n_pars
                else:
                    raise(NotImplementedError(f'Unknown type of xyz constraint for atom {atom["label"]}'))
            else:
                parameters = parameters.at[current_index:current_index + 3].set(list(xyz))
                xyz_instructions = Array(
                    parameter_tuple=tuple(RefinedParameter(par_index=int(array_index), multiplicator=1.0) for array_index in range(current_index, current_index + 3))
                )
                current_index += 3

            if atom['adp_type'] == 'Uani' or atom['adp_type'] == 'Umpe':
                adp = jnp.array(atom[['U_11', 'U_22', 'U_33', 'U_23', 'U_13', 'U_12']].values.astype(np.float64))
                if atom['label'] in constraint_dict.keys() and 'uij' in constraint_dict[atom['label']].keys():
                    constraint = constraint_dict[atom['label']]['uij']
                    if type(constraint).__name__ == 'ConstrainedValues':
                        instr_zip = zip(constraint.variable_indexes, constraint.multiplicators, constraint.added_value) 
                        adp_pars = Array(parameter_tuple=tuple(constrained_values_to_instruction(
                            par_index,
                            mult,
                            add,
                            constraint,
                            current_index) for par_index, mult, add in instr_zip)
                        )
                        adp_instructions = AnisoTFactor(uij_pars=adp_pars)
                        # we need this construction to unpack lists in indexes for the MultiIndexParameters
                        n_pars = max(max(entry) if is_multientry(entry) else entry for entry in constraint.variable_indexes) + 1

                        # MultiIndexParameter can never be unique so we can throw it out
                        u_indexes = [-1 if is_multientry(entry) else entry for entry in constraint.variable_indexes]

                        parameters = parameters.at[current_index:current_index + n_pars].set(
                            [adp[jnp.array(varindex)] for index, varindex in zip(*np.unique(u_indexes, return_index=True)) if index >=0]
                        )
                        current_index += n_pars
                    elif type(constraint).__name__ == 'UEquivConstraint':
                        bound_index = names.index(constraint.bound_atom_name)
                        adp_instructions = UEquivTFactor(parent_index=bound_index, scaling_par=FixedParameter(value=constraint.multiplicator))
                    else:
                        raise NotImplementedError('Unknown Uij Constraint')
                else:
                    parameters = parameters.at[current_index:current_index + 6].set(list(adp))
                    adp_pars = Array(
                        parameter_tuple=tuple(RefinedParameter(
                            par_index=int(array_index), multiplicator=1.0
                            ) for array_index in range(current_index, current_index + 6))
                    )
                    adp_instructions = AnisoTFactor(uij_pars=adp_pars)
                    current_index += 6
            elif atom['adp_type'] == 'Uiso':
                if atom['label'] in constraint_dict.keys() and 'uij' in constraint_dict[atom['label']].keys():
                    constraint = constraint_dict[atom['label']]['uij']
                    if type(constraint).__name__ == 'UEquivConstraint':
                            bound_index = names.index(constraint.bound_atom_name)
                            adp_instructions = UEquivTFactor(parent_index=bound_index, scaling_par=FixedParameter(value=constraint.multiplicator))
                else:
                    adp_par = RefinedParameter(par_index=int(current_index), multiplicator=1.0)
                    adp_instructions = IsoTFactor(uiso_par=adp_par)
                    parameters = parameters.at[current_index].set(float(atom['U_iso_or_equiv']))
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
                cijk_instructions = Array(
                    parameter_tuple = tuple(constrained_values_to_instruction(par_index, mult, add, constraint, current_index) for par_index, mult, add in instr_zip)
                )
                # we need this construction to unpack lists in indexes for the MultiIndexParameters
                n_pars = max(max(entry) if is_multientry(entry) else entry for entry in constraint.variable_indexes) + 1

                # MultiIndexParameter can never be unique so we can throw it out
                u_indexes = [-1 if is_multientry(entry) else entry for entry in constraint.variable_indexes]

                parameters = parameters.at[current_index:current_index + n_pars].set(
                    [cijk[jnp.array(varindex)] for index, varindex in zip(*np.unique(u_indexes, return_index=True)) if index >=0]
                )

                current_index += n_pars
            elif atom['label'] in atoms_for_gc3:
                parameters = parameters.at[current_index:current_index + 10].set(list(cijk))
                cijk_instructions = Array(
                    parameter_tuple=tuple(RefinedParameter(par_index=int(array_index), multiplicator=1.0) for array_index in range(current_index, current_index + 10))
                )
                current_index += 10
            else:
                cijk_instructions = Array(
                    parameter_tuple=tuple(FixedParameter(value=0.0) for index in range(10))
                )

            if 'D_1111' in atom.keys():
                dijkl = jnp.array(atom[['D_1111', 'D_2222', 'D_3333', 'D_1112', 'D_1222', 'D_1113', 'D_1333', 'D_2223', 'D_2333', 'D_1122', 'D_1133', 'D_2233', 'D_1123', 'D_1223', 'D_1233']].values.astype(np.float64))
            else:
                dijkl = jnp.zeros(15)

            if atom['label'] in constraint_dict.keys() and 'dijkl' in constraint_dict[atom['label']].keys() and atom['label'] in atoms_for_gc4:
                constraint = constraint_dict[atom['label']]['dijkl']
                instr_zip = zip(constraint.variable_indexes, constraint.multiplicators, constraint.added_value) 
                dijkl_instructions = Array(
                    parameter_tuple=tuple(constrained_values_to_instruction(par_index, mult*1e-3, add, constraint, current_index) for par_index, mult, add in instr_zip)
                )
                # we need this construction to unpack lists in indexes for the MultiIndexParameters
                n_pars = max(max(entry) if is_multientry(entry) else entry for entry in constraint.variable_indexes) + 1
                # MultiIndexParameter can never be unique so we can throw it out
                u_indexes = [-1 if is_multientry(entry) else entry for entry in constraint.variable_indexes]
                parameters = parameters.at[current_index:current_index + n_pars].set(
                    [dijkl[jnp.array(varindex)]*1e3 for index, varindex in zip(*np.unique(u_indexes, return_index=True)) if index >=0]
                )
                current_index += n_pars
            elif atom['label'] in atoms_for_gc4:
                parameters = parameters.at[current_index:current_index + 15].set(list(dijkl*1e3))
                dijkl_instructions = Array(
                    parameter_tuple=tuple(RefinedParameter(par_index=int(array_index), multiplicator=1e-3) for array_index in range(current_index, current_index + 15))
                )
                current_index += 15
            else:
                dijkl_instructions = Array(
                    parameter_tuple=tuple(FixedParameter(value=0.0) for index in range(15))
                )

            if atom['label'] in constraint_dict.keys() and 'occ' in constraint_dict[atom['label']].keys():
                constraint = constraint_dict[atom['label']]['occ']
                if type(constraint).__name__ == 'ConstrainedValues':
                    occupancy = FixedParameter(value=float(constraint.added_value[0]), special_position=constraint.special_position)
                elif type(constraint).__name__ == 'CommonOccupancyParameter':
                    if constraint.label in common_occupancy_indexes:
                        parameter_index = common_occupancy_indexes[constraint.label]
                    else:
                        parameter_index = current_index
                        common_occupancy_indexes[constraint.label] = parameter_index
                        parameters = parameters.at[current_index].set(atom['occupancy'] / float(constraint.multiplicator))
                        current_index += 1
                    occupancy = RefinedParameter(
                        par_index=int(parameter_index),
                        multiplicator=float(constraint.multiplicator),
                        added_value=float(constraint.added_value),
                        special_position=constraint.special_position
                    )
                else:
                    raise NotImplementedError('This type of constraint is not implemented for the occupancy')
            else:
                occupancy = FixedParameter(value=float(atom['occupancy']))

            construction_instructions[instruction_index] = AtomInstructions(
                name=atom['label'],
                element=atom['type_symbol'],
                dispersion_real = atom['type_scat_dispersion_real'],
                dispersion_imag = atom['type_scat_dispersion_imag'],
                xyz=xyz_instructions,
                uij=adp_instructions,
                cijk=cijk_instructions,
                dijkl=dijkl_instructions,
                occupancy=occupancy
            )
        if all(instr is not None for instr in construction_instructions):
            # there are no more dependencies
            break
    else:
        print(construction_instructions)
        raise ValueError('Could not build construction instructions, make sure that constrained values are based on refined values and not dependend onto each other.')
    parameters = parameters[:current_index]
    return tuple(construction_instructions), parameters
