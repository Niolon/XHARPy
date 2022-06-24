from .common import jnp
from .common import AtomInstructions
from ..conversion import ucif2ucart, cell_constants_to_M

from typing import List, Tuple
import numpy as np

def construct_values(
    parameters: jnp.ndarray,
    construction_instructions: List[AtomInstructions],
    cell_mat_m: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reconstruct xyz, adp-parameters, Gram-Charlier parameters and occupancies
    from the given construction instructions. Allows for the flexible usage of
    combinations of fixed parameters and parameters that are refined, as well as
    constraints

    Parameters
    ----------
    parameters : jnp.ndarray
        parameters used during the refinement
    construction_instructions : List[AtomInstructions]
        List of instructions for reconstructing the atomic parameters
    cell_mat_m : jnp.ndarray
        size (3, 3) array with the cell vectors as row vectors, used for Uiso
        calculation.

    Returns
    -------
    xyz: jnp.ndarray
        size (N,3) array of fractional coordinates for the atoms in the asymmetric
        unit
    uij: jnp.ndarray
        size (N, 6) array of anisotropic displacement parameters (isotropic
        parameters are transformed to anitropic parameters). Parameters are in
        the convention as used e.g. Shelxl or the cif as U. Order: U11, U22,
        U33, U23, U13, U12
    cijk: jnp.ndarray
        size (N, 10) array of third-order Gram-Charlier parameters as defined in 
        Inter. Tables of Cryst. B (2010): Eq 1.2.12.7. Order: C111, C222, C333,
        C112, C122, C113, C133, C223, C233, C123
    dijkl: jnp.ndarray
        size (N, 15) array of fourth-order Gram-Charlier parameters as defined in 
        Inter. Tables of Cryst. B (2010): Eq 1.2.12.7. Order: D1111, D2222,
        D3333, D1112, D1222, D1113, D_1333, D2223, D2333, D1122, D1133, D2233,
        D1123, D1223, D1233    
    occupancies: jnp.ndarray
        size (N) array of atomic occupancies. Atoms on special positions have an
        occupancy of 1/multiplicity
    """
    cell_mat_f = jnp.linalg.inv(cell_mat_m).T
    lengths_star = jnp.linalg.norm(cell_mat_f, axis=0)
    xyz = jnp.array(
        [instr.xyz.resolve(parameters) if not instr.xyz.derived 
        else jnp.full(3, jnp.nan) for instr in construction_instructions]
         
    )

    uij = jnp.array(
        [jnp.array([inner_instruction.resolve(parameters) for inner_instruction in instruction.uij])
          if type(instruction.uij) in (tuple, list) else jnp.full(6, -9999.9) for instruction in construction_instructions]
    )
    
    cijk = jnp.array(
        tuple(instruction.cijk.resolve(parameters) for instruction in construction_instructions)
    )
    
    dijkl = jnp.array(
        tuple(instruction.dijkl.resolve(parameters) for instruction in construction_instructions)
    )
    occupancies = jnp.array([instruction.occupancy.resolve(parameters) for instruction in construction_instructions])    

    # second loop here for constructed options in order to have everything already available
    for index, instruction in enumerate(construction_instructions):
        # constrained displacements
        if instruction.xyz.derived:
            xyz = xyz.at[index, :].set(instruction.xyz.resolve(parameters, xyz, cell_mat_m))
        if type(instruction.uij).__name__ == 'UEquivCalculated':
            uij_parent = uij[instruction.uij.atom_index, jnp.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])]
            u_cart = ucif2ucart(cell_mat_m, uij_parent[None,:, :])
            uiso = jnp.trace(u_cart[0]) / 3
            uij = uij.at[index, :3].set(jnp.array([uiso, uiso, uiso]))
            uij = uij.at[index, 3].set(uiso * jnp.sum(cell_mat_f[:, 1] * cell_mat_f[:, 2]) / lengths_star[1] / lengths_star[2])
            uij = uij.at[index, 4].set(uiso * jnp.sum(cell_mat_f[:, 0] * cell_mat_f[:, 2]) / lengths_star[0] / lengths_star[2])
            uij = uij.at[index, 5].set(uiso * jnp.sum(cell_mat_f[:, 0] * cell_mat_f[:, 1]) / lengths_star[0] / lengths_star[1])
        elif type(instruction.uij).__name__ == 'Uiso':
            uiso = instruction.uij.uiso.resolve(parameters)
            uij = uij.at[index, :3].set(jnp.array([uiso, uiso, uiso]))
            uij = uij.at[index, 3].set(uiso * jnp.sum(cell_mat_f[:, 1] * cell_mat_f[:, 2]) / lengths_star[1] / lengths_star[2])
            uij = uij.at[index, 4].set(uiso * jnp.sum(cell_mat_f[:, 0] * cell_mat_f[:, 2]) / lengths_star[0] / lengths_star[2])
            uij = uij.at[index, 5].set(uiso * jnp.sum(cell_mat_f[:, 0] * cell_mat_f[:, 1]) / lengths_star[0] / lengths_star[1])
    return xyz, uij, cijk, dijkl, occupancies


def construct_esds(
    var_cov_mat: jnp.ndarray,
    construction_instructions: List[AtomInstructions]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate the estimated standard deviations for the calculated values
    Will currently not resolve constraints, but might in the future.

    Parameters
    ----------
    var_cov_mat : jnp.ndarray
        size (P, P) array containing the variances and covariances, where P is the
        number of refined parameters.
    construction_instructions : List[AtomInstructions]
        List of instructions for reconstructing the atomic parameters

    Returns
    -------
    xyz_esd: jnp.ndarray
        size (N, 3) array containing floats for valid and np.nan for fractional 
        positions where there are no estimated deviations. N is the number
        of atoms in the asymmetric unit
    uij_esd: jnp.ndarray
        size (N, 6) array of esds of anisotropic displacement parameters
    cijkl_esd: jnp.ndarray
        size (N, 10) array of esds of third-order Gram-Charlier parameters
    dijkl_esd: jnp.ndarray
        size (N, 15) array of esds of third-order Gram-Charlier parameters
    occ_esd: jnp.ndarray
        size (N) array of esds of occupacions
    """
    xyz = jnp.array(
        [instruction.xyz.resolve_esd(var_cov_mat) for instruction in construction_instructions]
    )
    uij = jnp.array(
        [[inner_instruction.resolve_esd(var_cov_mat) for inner_instruction in instruction.uij]
          if type(instruction.uij) in (tuple, list, np.ndarray, jnp.ndarray) else jnp.full(6, jnp.nan) 
          for instruction in construction_instructions]
    )
    
    cijk = jnp.array(
        tuple(instruction.cijk.resolve_esd(var_cov_mat) for instruction in construction_instructions)
    )
    
    dijkl = jnp.array(
        tuple(instruction.dijkl.resolve_esd(var_cov_mat) for instruction in construction_instructions)
    )
    occupancies = jnp.array([instruction.occupancy.resolve_esd(var_cov_mat) for instruction in construction_instructions])
    return xyz, uij, cijk, dijkl, occupancies    


def distance_with_esd(
    atom1_name: str,
    atom2_name: str,
    construction_instructions: List[AtomInstructions],
    parameters: jnp.ndarray,
    var_cov_mat: jnp.ndarray,
    cell_par: jnp.ndarray,
    cell_esd: jnp.ndarray,
    crystal_system: str,
    symm_mat2: jnp.ndarray = jnp.eye(3),
    symm_vec2: jnp.ndarray = np.zeros(3)
)-> Tuple[float, float]:
    """Calculates the distance value of a given atom pair and its estimated
    standart deviation

    Parameters
    ----------
    atom1_name : str
        Label of the first atom
    atom2_name : str
        label of the second atom
    construction_instructions : List[AtomInstructions]
        List of atomic instruction for reconstruction of the parameters. 
        Needs to be the same, that was used for refinement.
    parameters : jnp.ndarray
        size (P) array of refined parameters
    var_cov_mat : jnp.ndarray
        size (P, P) array of the variance-covariance matrix
    cell_par : jnp.ndarray
        size (6) array of cell parameters in degrees and Angstroem. Depending
        on the given crystal system only the first of equivalent values might
        be used. If the angle has to be 90° the angle values will be ignored
    cell_esd : jnp.ndarray
        size (6) array of the estimated standard deviation of cell parameters
        Only certain values might be used, see cell_par
    crystal_system : str
        Crystal system of the evaluated structure. Possible values are: 
        'triclinic', 'monoclinic' 'orthorhombic', 'tetragonal', 'hexagonal',
        'trigonal' and 'cubic'. Is considered for the esd calculation.
    symm_mat2: jnp.ndarray, optional
        size (3, 3) symmetry matrix to convert the coordinate of atom2, defaults to 
        jnp.eye(3)
    symm_vec2: jnp.ndarray, optional
        size (3) array containing the translation vector for atom2,
        defaults to jnp.zeros(3)
    Returns
    -------
    distance: float
        Calculated distance between the two atoms
    distance_esd: float
        Estimated standard deviation of the calculated distance
    """
    names = [instr.name for instr in construction_instructions]
    index1 = names.index(atom1_name)
    index2 = names.index(atom2_name)

    def distance_func(parameters, cell_par):
        cell_mat_m = cell_constants_to_M(*cell_par, crystal_system)
        constructed_xyz, *_ = construct_values(parameters, construction_instructions, cell_mat_m)
        coord1 = constructed_xyz[index1]
        coord2 = symm_mat2 @ constructed_xyz[index2] + symm_vec2

        return jnp.linalg.norm(cell_mat_m @ (coord1 - coord2))
    
    distance = distance_func(parameters, cell_par)

    jac1, jac2 = jax.grad(distance_func, [0, 1])(parameters, cell_par)

    esd = jnp.sqrt(jac1[None, :] @ var_cov_mat @ jac1[None, :].T + jac2[None,:] @ jnp.diag(cell_esd**2) @ jac2[None,:].T)
    return distance, esd[0, 0]


def u_iso_with_esd(
    atom_name: str,
    construction_instructions: List[AtomInstructions],
    parameters: jnp.ndarray,
    var_cov_mat: jnp.ndarray,
    cell_par: jnp.ndarray,
    cell_esd: jnp.ndarray,
    crystal_system: str
)-> Tuple[float, float]:
    """Calculate the Uequiv value from anisotropic displacement parameters for
    a given atom.

    Parameters
    ----------
    atom_name : str
        Label of the atom
    construction_instructions : List[AtomInstructions]
        List of atomic instruction for reconstruction of the parameters. 
        Needs to be the same, that was used for refinement.
    parameters : jnp.ndarray
        size (P) array of refined parameters
    var_cov_mat : jnp.ndarray
        size (P, P) array of the variance-covariance matrix
    cell_par : jnp.ndarray
        size (6) array of cell parameters in degrees and Angstroem. Depending
        on the given crystal system only the first of equivalent values might
        be used. If the angle has to be 90° the angle values will be ignored
    cell_esd : jnp.ndarray
        size (6) array of the estimated standard deviation of cell parameters
        Only certain values might be used, see cell_par
    crystal_system : str
        Crystal system of the evaluated structure. Possible values are: 
        'triclinic', 'monoclinic' 'orthorhombic', 'tetragonal', 'hexagonal',
        'trigonal' and 'cubic'. Is considered for the esd calculation.

    Returns
    -------
    u_equiv: float
        Calculated U(equiv) value
    u_equiv_esd, float
        Estimated standard deviation of the U(equiv) value
    """
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
    esd = jnp.sqrt(jac1[None, :] @ var_cov_mat @ jac1[None, :].T + jac2[None,:] @ jnp.diag(cell_esd**2) @ jac2[None,:].T)
    return u_iso, esd[0, 0]


def angle_with_esd(
    atom1_name: str,
    atom2_name: str,
    atom3_name: str, 
    construction_instructions: List[AtomInstructions],
    parameters: jnp.ndarray,
    var_cov_mat: jnp.ndarray,
    cell_par: jnp.ndarray,
    cell_esd: jnp.ndarray,
    crystal_system: str,
    symm_mat1: jnp.ndarray = jnp.eye(3),
    symm_vec1: jnp.ndarray = np.zeros(3),
    symm_mat3: jnp.ndarray = jnp.eye(3),
    symm_vec3: jnp.ndarray = np.zeros(3)
)-> Tuple[float, float]:
    """Calculates the angle with its estimated standard deviation spanned by
    atom1-atom2-atom3

    Parameters
    ----------
    atom1_name : str
        Label of the first outer atom of the angle
    atom2_name : str
        Label of the central atom of the angle
    atom3_name : str
        Label of the second outer atom of the angle
    construction_instructions : List[AtomInstructions]
        List of atomic instruction for reconstruction of the parameters. 
        Needs to be the same, that was used for refinement.
    parameters : jnp.ndarray
        size (P) array of refined parameters
    var_cov_mat : jnp.ndarray
        size (P, P) array of the variance-covariance matrix
    cell_par : jnp.ndarray
        size (6) array of cell parameters in degrees and Angstroem. Depending
        on the given crystal system only the first of equivalent values might
        be used. If the angle has to be 90° the angle values will be ignored
    cell_esd : jnp.ndarray
        size (6) array of the estimated standard deviation of cell parameters
        Only certain values might be used, see cell_par
    crystal_system : str
        Crystal system of the evaluated structure. Possible values are: 
        'triclinic', 'monoclinic' 'orthorhombic', 'tetragonal', 'hexagonal',
        'trigonal' and 'cubic'. Is considered for the esd calculation.
    symm_mat2: jnp.ndarray, optional
        size (3, 3) symmetry matrix to convert the coordinate of atom2, defaults to 
        jnp.eye(3)
    symm_vec2: jnp.ndarray, optional
        size (3) array containing the translation vector for atom2,
        defaults to jnp.zeros(3)
    symm_mat3: jnp.ndarray, optional
        size (3, 3) symmetry matrix to convert the coordinate of atom3, defaults to 
        jnp.eye(3)
    symm_vec3: jnp.ndarray, optional
        size (3) array containing the translation vector for atom3,
        defaults to jnp.zeros(3)

    Returns
    -------
    angle: float
        calculated angle between the three atoms
    angle_esd: float
        estimated standard deviation for the calculated angle
    """
    names = [instr.name for instr in construction_instructions]
    index1 = names.index(atom1_name)
    index2 = names.index(atom2_name)
    index3 = names.index(atom3_name)

    def angle_func(parameters, cell_par):
        cell_mat_m = cell_constants_to_M(*cell_par, crystal_system)
        constructed_xyz, *_ = construct_values(parameters, construction_instructions, cell_mat_m)
        xyz1 = symm_mat1 @ constructed_xyz[index1] + symm_vec1
        xyz2 = constructed_xyz[index2] 
        xyz3 = symm_mat3 @ constructed_xyz[index3] + symm_vec3
        vec1 = cell_mat_m @ (xyz1 - xyz2)
        vec2 = cell_mat_m @ (xyz3 - xyz2)

        return jnp.rad2deg(jnp.arccos((vec1 / jnp.linalg.norm(vec1)) @ (vec2 / jnp.linalg.norm(vec2))))
    
    angle = angle_func(parameters, cell_par)

    jac1, jac2 = jax.grad(angle_func, [0, 1])(parameters, cell_par)

    esd = (jnp.sqrt(jac1[None, :] @ var_cov_mat @ jac1[None, :].T 
           + jac2[None,:] @ jnp.diag(cell_esd**2) @ jac2[None,:].T))
    return angle, esd[0, 0]