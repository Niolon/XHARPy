from collections import OrderedDict
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax
import re
from .xharpy import ConstrainedValues


def ciflike_to_dict(filename, return_only_loops=False, resolve_std=True):
    """
    Takes the filename of a ciflike file such as the cif format itself or fco files
    and returns the content as an ordered dictionary.
    
    All tables, that are contained in the loop(ed) sections are returned under the 
    'loops' keyword as pandas dataframes. Everything else is stored under the keyword
    in cif minus the leading underscore. Will work with multiple structures in one cif.
    That is why the first dict has data names as keys and the actual other keywords as
    a contained dict.
    
    Has two keyword options:
    return_only_loops: [False] will only return the contained loops
    resolve_std: [True] will introduce new keywords with _std for all values containing
                 errors such as 11(2). Also works on the dataframes.
    
    """
    PATTERN = re.compile(r'''((?:[^ "']|"[^"]*"|'[^']*')+)''')
    with open(filename, 'r') as fo:
        lines = [line[:-1] for line in fo.readlines()]
    datablocks = OrderedDict()
    current_loop_lines = []
    current_loop_titles = []
    current_block = 'preblock'
    in_loop = False
    in_loop_titles = False
    current_line_collect = []
    for index, raw_line in enumerate(lines):
        line = raw_line.strip().lstrip()
        if line == '' or index == len(lines) - 1:
            if in_loop:
                in_loop = False
                new_df = pd.DataFrame(current_loop_lines)
                for key in new_df:
                    new_df[key] = pd.to_numeric(new_df[key], errors='ignore')
                if resolve_std:
                    for column in new_df.columns:
                        if new_df[column].dtype != 'O':
                            continue
                        concatenate = ''.join(new_df[column])
                        if  re.search(r'[\(\)]', concatenate) is not None and re.search(r'[^\d^\.^\(^\)\-\+]', concatenate) is None:
                            values, errors = np.array([split_error(val) for val in new_df[column]]).T
                            new_df[column] = values
                            new_df[column+'_std'] = errors
                datablocks[current_block]['loops'].append(new_df)
                current_loop_lines = []
                current_loop_titles = []
                current_line_collect = []
            continue
        if in_loop and not in_loop_titles and line.startswith('_') or line.startswith('loop_'):
            in_loop = False
            if len(current_loop_lines) > 0:
                new_df = pd.DataFrame(current_loop_lines)
                for key in new_df:
                    new_df[key] = pd.to_numeric(new_df[key], errors='ignore')
                if resolve_std:
                    for column in new_df.columns:
                        if new_df[column].dtype != 'O':
                            continue
                        concatenate = ''.join(new_df[column])
                        if  re.search(r'[\(\)]', concatenate) is not None and re.search(r'[^\d^\.^\(^\)\-\+]', concatenate) is None:
                            values, errors = np.array([split_error(val) for val in new_df[column]]).T
                            new_df[column] = values
                            new_df[column+'_std'] = errors
                datablocks[current_block]['loops'].append(new_df)
            current_loop_lines = []
            current_loop_titles = []
            current_line_collect = []
        if line[0] == '#':
            continue
        elif line[:5] == 'data_':
            current_block = line[5:]
            datablocks[current_block] = OrderedDict([('loops', [])])
        elif line[:5] == 'loop_':
            in_loop = True
            in_loop_titles = True
        elif in_loop and in_loop_titles and line[0] == '_':
            current_loop_titles.append(line[1:])
        elif in_loop:
            in_loop_titles = False
            line_split = [item.strip() for item in PATTERN.split(line) if item != '' and not item.isspace()]
            line_split = [item[1:-1] if "'" in item else item for item in line_split]
            current_line_collect += line_split
            if len(current_line_collect) == len(current_loop_titles):
                current_loop_lines.append(OrderedDict())
                for index2, item in enumerate(current_line_collect):
                    current_loop_lines[-1][current_loop_titles[index2]] = item
                current_line_collect = []
        elif line[0] == '_':
            line_split = [item.strip() for item in PATTERN.split(line) if item != '' and not item.isspace()]
            line_split = [item[1:-1] if "'" in item else item for item in line_split]
            if len(line_split) > 1:
                if resolve_std:
                    test = line_split[1]
                    if re.search(r'[^\d]', test) is None:
                        datablocks[current_block][line_split[0][1:]] = int(test)
                    elif re.search(r'[^\d^\.]', test) is None and re.search(r'\d', test) is not None:
                        datablocks[current_block][line_split[0][1:]] = float(test)
                    elif re.search(r'[\(\)]', test) is not None and re.search(r'[^\d^\.^\(^\)\-\+]', test) is None:
                        val, error = split_error(test)
                        datablocks[current_block][line_split[0][1:]] = val
                        datablocks[current_block][line_split[0][1:] + '_std'] = error
                    else:
                        datablocks[current_block][line_split[0][1:]] = line_split[1]
                else:
                    datablocks[current_block][line_split[0][1:]] = line_split[1]
    if return_only_loops:
        loops = []
        for value in datablocks.values():
            for loop in value['loops']:
                loops.append(loop)
        return loops
    else:
        return datablocks

def split_error(string):
    """
    Helper function to split a string containing a value with error in brackets
    to a single value.
    """
    int_search = re.search(r'([\-\d]*)\((\d*)\)', string)
    search = re.search(r'(\-{0,1})([\d]*)\.(\d*)\((\d*)\)', string)
    if search is not None:
        sign, before_dot, after_dot, err = search.groups()
        if sign == '-':
            return -1 * (int(before_dot) + int(after_dot) * 10**(-len(after_dot))), int(err) * 10**(-len(after_dot))
        else:
            return int(before_dot) + int(after_dot) * 10**(-len(after_dot)), int(err) * 10**(-len(after_dot))
    elif int_search is not None:
        value, error = int_search.groups()
        return int(value), int(error)
    else:
        return float(string), 0.0  


def symm_to_matrix_vector(instruction):
    """
    Converts a instruction such as -x, -y, 0.5+z to a symmetry matrix and a 
    translation vector
    """
    instruction_strings = [val.replace(' ', '').upper() for val in instruction.split(',')]
    matrix = jnp.zeros((3,3), dtype=np.float64)
    vector = jnp.zeros(3, dtype=np.float64)
    for xyz, element in enumerate(instruction_strings):
        # search for fraction in a/b notation
        fraction1 = re.search(r'(-{0,1}\d{1,3})/(\d{1,3})(?![XYZ])', element)
        # search for fraction in 0.0 notation
        fraction2 = re.search(r'(-{0,1}\d{0,1}\.\d{1,4})(?![XYZ])', element)
        # search for whole numbers
        fraction3 = re.search(r'(-{0,1}\d)(?![XYZ])', element)
        if fraction1:
            vector = jax.ops.index_update(vector, jax.ops.index[xyz], float(fraction1.group(1)) / float(fraction1.group(2)))
        elif fraction2:
            vector = jax.ops.index_update(vector, jax.ops.index[xyz], float(fraction2.group(1)))
        elif fraction3:
            vector = jax.ops.index_update(vector, jax.ops.index[xyz], float(fraction3.group(1)))

        symm = re.findall(r'-{0,1}[\d\.]{0,8}[XYZ]', element)
        for xyz_match in symm:
            if len(xyz_match) == 1:
                sign = 1
            elif xyz_match[0] == '-' and len(xyz_match) == 2:
                sign = -1
            else:
                sign = float(xyz_match[:-1])
            if xyz_match[-1] == 'X':
                matrix = jax.ops.index_update(matrix, jax.ops.index[xyz, 0], sign)
            if xyz_match[-1] == 'Y':
                matrix = jax.ops.index_update(matrix, jax.ops.index[xyz, 1], sign)
            if xyz_match[-1] == 'Z':
                matrix = jax.ops.index_update(matrix, jax.ops.index[xyz, 2], sign)
    return matrix, vector


def cif2data(cif_name, cif_dataset=0):
    cif_raw = ciflike_to_dict(cif_name)
    if type(cif_dataset) is int:
        cif = cif_raw[list(cif_raw.keys())[cif_dataset]]
    elif type(cif_dataset) is str:
        cif = cif_raw[cif_dataset]
    else:
        raise ValueError('Invalid cif_dataset value. Must be either index as int or name as str')
    cell = np.array([cif['cell_length_a'],
                     cif['cell_length_b'],
                     cif['cell_length_c'],
                     cif['cell_angle_alpha'],
                     cif['cell_angle_beta'],
                     cif['cell_angle_gamma']])


    std_keys = ['cell_length_a_std', 'cell_length_b_std', 'cell_length_c_std',
                'cell_angle_alpha_std', 'cell_angle_beta_std', 'cell_angle_gamma_std']

    cell_std = np.array([cif[key] if key in list(cif.keys()) else 0.0 for key in std_keys])

    atom_table = [table for table in cif['loops'] if 'atom_site_label' in table.columns][0].copy()
    atom_table.columns = [label.replace('atom_site_', '') for label in atom_table.columns]

    adp_table = [table for table in cif['loops'] if 'atom_site_aniso_label' in table.columns][0].copy()
    adp_table.columns = [label.replace('atom_site_aniso_', '') for label in adp_table.columns]

    disp_corr_table = [table for table in cif['loops'] if 'atom_type_scat_dispersion_real' in table.columns][0].copy()
    disp_corr_table.columns = [label.replace('atom_', '') for label in disp_corr_table.columns]

    atom_table = pd.merge(atom_table, adp_table, on='label', how='left').copy() # put adp parameters into table
    atom_table = pd.merge(atom_table, disp_corr_table, on='type_symbol', how='left') # add f' and f'' parameters

    #cell_mat_g_star = np.einsum('ja, jb -> ab', cell_mat_f, cell_mat_f)
    symmetry_table = [table for table in cif['loops'] if 'space_group_symop_operation_xyz' in table.columns or 'symmetry_equiv_pos_as_xyz' in table.columns][0].copy()
    symmetry_table = symmetry_table.rename({'symmetry_equiv_pos_as_xyz': 'space_group_symop_operation_xyz'}, axis=1)
    symm_list = [symm_to_matrix_vector(instruction) for instruction in symmetry_table['space_group_symop_operation_xyz'].values]
    symm_mats_r, symm_vecs_t = zip(*symm_list) # basically transposes the lists
    symm_mats_r = np.array(symm_mats_r)
    symm_vecs_t = np.array(symm_vecs_t)
    symm_mats_vecs = (symm_mats_r, symm_vecs_t)
    return atom_table, cell, cell_std, symm_mats_vecs, cif['diffrn_radiation_wavelength']



def instructions_to_constraints(names, instructions):
    variable_indexes = list(range(len(names)))
    multiplicators = [1.0] * len(names)
    added_value = [0.0] * len(names)
    for index, name in enumerate(names):
        if name not in instructions:
            continue
        for following_index in range(index + 1, len(names)):
            variable_indexes[following_index] -= 1
        mult, var, add = instructions[name]
        if var == '':
            variable_indexes[index] = -1
            multiplicators[index] = 0.0
            added_value[index] = add
        else:
            variable_indexes[index] = variable_indexes[names.index(var)]
            multiplicators[index] = mult
            added_value[index] = add
    return ConstrainedValues(variable_indexes=variable_indexes,
                             multiplicators=multiplicators,
                             added_value=added_value) 



def lst2constraint_dict(filename):
    with open(filename) as fo:
        lst_content = fo.read()

    find = re.search(r'Special position constraints.*?\n\n\n', lst_content, flags=re.DOTALL)  

    xyz_names = ['x', 'y', 'z']
    uij_names = ['U11', 'U22', 'U33', 'U23', 'U13', 'U12']
    occ_names = ['sof']
    names = xyz_names + uij_names + occ_names

    constraint_dict = {}
    for entry in find.group(0).strip().split('\n\n'):
        lines = entry.split('\n')
        name = lines[0].split()[-1]
        instructions = {}
        for line in lines[1:-1]:
            for op in line.strip().split('   '):
                add = 0.0
                mult = 0.0
                var = ''
                target, instruction = [e.strip() for e in op.split(' = ')]
                for sum_part in instruction.split(' + '):
                    sum_part = sum_part.strip()
                    if '*' in sum_part:
                        for prod_part in sum_part.split(' * '):
                            prod_part = prod_part.strip()
                            if prod_part in names:
                                var = prod_part
                            else:
                                mult = float(prod_part)
                    else:
                        if sum_part in names:
                            var = sum_part
                            mult = 1
                        add = float(sum_part)
                instructions[target] = (mult, var, add)
        constraint_dict[name] = {
            'xyz': instructions_to_constraints(xyz_names, instructions),
            'uij': instructions_to_constraints(uij_names, instructions),
            'occ': instructions_to_constraints(occ_names, instructions)
        }
    return constraint_dict