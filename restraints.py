from collections import namedtuple
import jax.numpy as np
import pandas as pd
from .conversion import ucif2ucart


#### User Creatable Objects

FixedDistanceRestraint = namedtuple('FixedDistanceRestraint', [
    'atom1', # Name of atom1 in the Distance Restraint
    'atom2', # Name of atom2 in the Distance Restraint
    'distance', # Distance in Angstrom
    'stderr' # sigma**2 of the restraint
])

SameDistanceRestraint = namedtuple('SameDistanceRestraint', [
    'atom_pairs', # list of tuples of atom pair names
    'stderr' # sigma**2 of the restraint
])

SameADPRestraint = namedtuple('SameADPRestraint', [
    'atom1', # Name of atom1 in the ADP Restraint
    'atom2', # Name of atom2 in the ADP Restraint
    'stderr', # sigma**2 of the restraint
    'mult' # multiplicator for ADP of atom2 (For H)
], defaults=[None, None, None, 1.0])

RigidADPRestraint = namedtuple('RigidADPRestraint', [
    'atom1', # Name of atom1 in the ADP Restraint
    'atom2', # Name of atom2 in the ADP Restraint
    'stderr', # sigma**2 of the restraint
    'mult' # multiplicator for ADP of atom2 (For H)
], defaults=[None, None, None, 1.0])


### Internal Objects
FixedDistanceRestrInd = namedtuple('DistanceRestrInd', [
    'atom1_index', # Index of atom1 in the distance restraint
    'atom2_index', # Index of atom2 in the distance restraint
    'distance', # Distance in angstroms
    'stderr' # sigma**2 of the restraint
])

SameDistanceRestrInd = namedtuple('SameDistanceRestrInd', [
    'atom_pair_indexes', # list of tuples of atom pair indexes
    'stderr' # sigma**2 of the restraint
])

SameADPRestrInd = namedtuple('SameADPRestrInd', [
    'atom1_index', # Index of atom1 in the ADP restraint
    'atom2_index', # Index of atom2 in the ADP restraint
    'stderr', # sigma**2 of the restraint
    'mult' # multiplicator for ADP of atom2 (For H)
])

RigidADPRestrInd = namedtuple('RigidADPRestrInd', [
    'atom1_index', # Index of atom1 in the ADP restraint
    'atom2_index', # Index of atom2 in the ADP restraint
    'stderr', # sigma**2 of the restraint
    'mult' # multiplicator for ADP of atom2 (For H)
])

### Functions

def create_restraint_instructions(atom_table, restraints):
    # For the time being takes only explicit pairs. In the end shoud be modified to take ranges like SIMU in ShelXL
    # At the end maybe even do SAME
    return_list = []
    names = atom_table['label']
    for restraint in restraints:
        if type(restraint).__name__ == 'FixedDistanceRestraint':
            return_list.append(FixedDistanceRestrInd(
                atom1_index=np.where(names == restraint.atom1)[0][0],
                atom2_index=np.where(names == restraint.atom2)[0][0],
                distance=restraint.distance,
                stderr=restraint.stderr 
                ))
        elif type(restraint).__name__ == 'SameDistanceRestraint':
            return_list.append(SameDistanceRestrInd(
                atom_pair_indexes=[(np.where(names == atom1)[0][0], np.where(names == atom2)[0][0]) for atom1, atom2 in restraint.atom_pairs],
                stderr=restraint.stderr
            ))
        elif type(restraint).__name__ == 'SameADPRestraint':
            return_list.append(SameADPRestrInd(
                atom1_index=np.where(names == restraint.atom1)[0][0],
                atom2_index=np.where(names == restraint.atom2)[0][0],
                stderr=restraint.stderr,
                mult=restraint.mult
            ))
        elif type(restraint).__name__ == 'RigidADPRestraint':
            return_list.append(RigidADPRestrInd(
                atom1_index=np.where(names == restraint.atom1)[0][0],
                atom2_index=np.where(names == restraint.atom2)[0][0],
                stderr=restraint.stderr,
                mult=restraint.mult
            ))
    return return_list

def resolve_restraints(xyz, uij, restraints, cell_mat_m):
    return_sum = 0.0
    for restraint in restraints:
        if type(restraint).__name__ == 'FixedDistanceRestrInd':
            position1 = xyz[restraint.atom1_index]
            position2 = xyz[restraint.atom2_index]
            diff = cell_mat_m @ (position1 - position2)
            return_sum += (np.linalg.norm(diff) - restraint.distance)**2 / restraint.stderr**2
        elif type(restraint).__name__ == 'SameDistanceRestrInd':
            distances = np.array([np.linalg.norm(cell_mat_m @ (xyz[index1] - xyz[index2]))
                                  for index1, index2 in restraint.atom_pair_indexes])
            return_sum += np.sum((distances - distances.mean)**2) / restraint.stderr**2
        elif type(restraint).__name__ == 'SameADPRestrInd':
            return_sum += np.sum((uij[restraint.atom1_index] - restraint.mult * uij[restraint.atom2_index])**2) / restraint.stderr**2
        elif type(restraint).__name__ == 'RigidADPRestrInd':
            #1 Calculate Rotation
            diff = cell_mat_m @ (xyz[restraint.atom1_index] - xyz[restraint.atom2_index])
            norm_vector = diff / np.linalg.norm(diff)
            z = np.array([0.0, 0.0, 1.0])
            v = np.cross(norm_vector, z)
            s = np.sqrt(np.sum(v**2))
            c = np.dot(diff, z)
            v_x = np.array([[0,     -v[2], v[1]],
                            [v[2],  0,     -v[0]],
                            [-v[1], v[0],  0]])
            rot = np.identity(3) + v_x + v_x @ v_x * (1 - c) / s**2

            #2 Calculate U_carts
            uij_selected = uij[[restraint.atom1_index, restraint.atom2_index], :]
            u_cart1, u_cart2 = ucif2ucart(cell_mat_m, uij_selected[:, [[0, 5, 4], [5, 1, 3], [4, 3, 2]]])
            # = ucif2ucart(cell_mat_m, uij[, [[0, 5, 4], [5, 1, 3], [4, 3, 2]]])

            #3 Rotate both U_carts (R @ U_cart @ R.T)
            u_cart1_r = rot @ u_cart1 @ rot.T
            u_cart2_r = restraint.mult * rot @ u_cart2 @ rot.T

            #4 Calculate differences
            diff_u = u_cart1_r - u_cart2_r
            return_sum += (diff_u[2, 2]**2 + diff_u[1, 2]**2 + diff_u[0, 2]**2) / restraint.stderr**2

    return return_sum

