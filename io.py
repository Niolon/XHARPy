from collections import OrderedDict
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax
import re
import warnings
from .xharpy import ConstrainedValues, cell_constants_to_M, distance_with_esd, construct_esds, construct_values


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
        lines = [line for line in fo.readlines()]
    datablocks = OrderedDict()
    current_loop_lines = []
    current_loop_titles = []
    # If there is data before the first data entrie store it as preblock
    current_block = 'preblock'
    in_loop = False
    in_loop_titles = False
    in_multiline = False
    multiline_title = 'InvalidTitle' # This should never be used
    multiline_entries = []
    current_line_collect = []
    try:
        for index, raw_line in enumerate(lines):
            line = raw_line.strip().lstrip()
            if len(line.strip()) == 0 or line.startswith('#'):
                # empty or comment line
                continue
            if in_loop and not in_loop_titles and (line.startswith('_') or line.startswith('loop_')):
                # The current loop has ended append entries as new DataFrame
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
                # empty all stored entries
                current_loop_lines = []
                current_loop_titles = []
                current_line_collect = []
            if line.startswith('data_'):
                # New data block
                current_block = line[5:]
                if current_block not in datablocks:
                    datablocks[current_block] = OrderedDict([('loops', [])])
            elif line.startswith('loop_'):
                # a new loop / table starts
                in_loop = True
                in_loop_titles = True
            elif in_loop and in_loop_titles and line.startswith('_'):
                # This line is a title entry within a loop
                current_loop_titles.append(line[1:])
            elif in_loop:
                # This line contains data within a loop
                in_loop_titles = False
                line_split = [item.strip() for item in PATTERN.split(line) if item != '' and not item.isspace()]
                line_split = [item[1:-1] if "'" in item else item for item in line_split]
                current_line_collect += line_split
                if len(current_line_collect) == len(current_loop_titles):
                    current_loop_lines.append(OrderedDict())
                    for index2, item in enumerate(current_line_collect):
                        current_loop_lines[-1][current_loop_titles[index2]] = item
                    current_line_collect = []
            elif line.startswith('_'):
                # we are not in a loop -> single line or multiline string entry
                line_split = [item.strip() for item in PATTERN.split(line) if item != '' and not item.isspace()]
                line_split = [item[1:-1] if "'" in item else item for item in line_split]
                if len(line_split) > 1:
                    if resolve_std:
                        test = line_split[1]
                        if len(test) == 0:
                            datablocks[current_block][line_split[0][1:]] = None
                        elif re.search(r'[^\d]', test) is None:
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
                else:
                    multiline_title = line_split[0][1:]
            elif line.startswith(';') and in_multiline:
                datablocks[current_block][multiline_title] = '\n'.join(multiline_entries)
                multiline_entries = []
                in_multiline = False
            elif line.startswith(';') and not in_multiline:
                in_multiline = True
            elif in_multiline:
                multiline_entries.append(line)
    except Exception as e:
        print('Error in Line {index}')
        print(line)
        raise e

    
    # We might have a final loop
    if in_loop:
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
    if 'type_symbol' not in atom_table:
        atom_table['type_symbol'] = [str(re.match(r'([A-Za-z]{1,2})\d', line['label']).groups(1)[0]) for _, line in atom_table.iterrows()]

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
    symm_strings = list(symmetry_table['space_group_symop_operation_xyz'].values)

    atom_table = atom_table.rename({'thermal_displace_type': 'adp_type'}, axis=1).copy()
    return atom_table, cell, cell_std, symm_mats_vecs, symm_strings, cif['diffrn_radiation_wavelength']


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
    if find is None:
        return {}

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
                target, instruction = [e.strip() for e in op.split(' =')]
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


def write_fcf(filename, hkl, refine_dict, parameters, symm_strings, structure_factors, cell):
    cell_mat_m = cell_constants_to_M(*cell)
    hkl_out = hkl.copy()
    for index in ('h', 'k', 'l'):
        values = hkl_out[index].values.copy()
        values[values < 0] += 10000
        hkl_out[index + 'sort'] = values
    hkl_out['indexes'] = list(range(len(hkl_out)))
    hkl_out = hkl_out.sort_values(['l', 'k', 'h'])
    cell_mat_f = np.linalg.inv(cell_mat_m).T
    index_vec_h = hkl_out[['h', 'k', 'l']].values.copy()
    intensity = hkl_out['intensity'].values.copy()
    stderr = hkl_out['stderr'].values.copy()
    structure_factors = structure_factors[hkl_out['indexes']].copy()

    wavelength = refine_dict['wavelength']

    if refine_dict['core'] == 'scale':
        extinction_parameter = 2
    else:
        extinction_parameter = 1

    if refine_dict['extinction'] == 'none':
        f_calc = np.abs(structure_factors)
        intensity_fcf = intensity / parameters[0]
        stderr_fcf = stderr / parameters[0]
    else:
        i_calc0 = np.abs(structure_factors)**2
        if refine_dict['extinction'] == 'secondary':
            # Secondary exctinction, as shelxl needs a wavelength                
            f_calc = np.abs(structure_factors) * np.sqrt(parameters[0] / (1 + parameters[extinction_parameter] * i_calc0)) 
            intensity_fcf = intensity / parameters[0] * (1 + parameters[extinction_parameter] * i_calc0)
            stderr_fcf = stderr / parameters[0] * (1 + parameters[extinction_parameter] * i_calc0)
            
        else:
            sintheta = np.linalg.norm(np.einsum('xy, zy -> zx', cell_mat_f, index_vec_h), axis=1) / 2 * wavelength
            sintwotheta = 2 * sintheta * np.sqrt(1 - sintheta**2)
            extinction_factors = 0.001 * wavelength**3 / sintwotheta
            f_calc = np.abs(structure_factors) * np.sqrt(parameters[0] / np.sqrt(1 + parameters[extinction_parameter] * extinction_factors * i_calc0)) 
            intensity_fcf = intensity / parameters[0] * np.sqrt(1 + parameters[extinction_parameter] * extinction_factors * i_calc0)
            stderr_fcf = stderr / parameters[0] * np.sqrt(1 + parameters[extinction_parameter] * extinction_factors * i_calc0)
            
    symm_string = "'" + "'\n'".join(symm_strings) + "'"

    angles = np.rad2deg(np.angle(structure_factors)) % 360

    header = f"""#
# h,k,l, Fo-squared, sigma(Fo-squared), Fc and phi(calc)
#
data_har
_shelx_refln_list_code          6
_shelx_F_calc_maximum      {np.max(f_calc):6.2f}

loop_
    _space_group_symop_operation_xyz
{symm_string}

_cell_length_a     {cell[0]:6.4f}
_cell_length_b     {cell[1]:6.4f}
_cell_length_c     {cell[2]:6.4f}
_cell_angle_alpha  {cell[3]:6.4f}
_cell_angle_beta   {cell[4]:6.4f}
_cell_angle_gamma  {cell[5]:6.4f}

loop_
    _refln_index_h
    _refln_index_k
    _refln_index_l
    _refln_F_squared_meas
    _refln_F_squared_sigma
    _refln_F_calc
    _refln_phase_calc
"""

    lines = ''.join([f'{h} {k} {l} {out_inten:.2f} {out_std:.2f} {norm_val:.2f} {angle:.1f}\n' for (h, k, l), out_inten, out_std, norm_val, angle in zip(index_vec_h, intensity_fcf, stderr_fcf, f_calc, angles)])

    with open(filename, 'w') as fo:
        fo.write(header + lines)


def entries2atom_string(entries):
    strings = [
        entries['label'],
        str(entries['sfac_index']),
        '{:8.6f}'.format((entries['fract_x'])),
        '{:8.6f}'.format((entries['fract_y'])),
        '{:8.6f}'.format((entries['fract_z'])),
        '{:8.5f}'.format((entries['occupancy'] + 10)),
        '{:7.5f}'.format((entries['U_11'])),
        '{:7.5f}'.format((entries['U_22'])),
        '{:7.5f}'.format((entries['U_33'])),
        '{:7.5f}'.format((entries['U_23'])),
        '{:7.5f}'.format((entries['U_13'])),
        '{:7.5f}'.format((entries['U_12']))
    ]

    atom_string = ''
    total = 0
    for string in strings:
        if total + len(string) + 1 < 70:
            atom_string += string + ' '
            total += len(string) + 1
        else:
            atom_string += ' =\n   ' + string + ' '
            total = 4 + len(string)
    return atom_string


def write_res(out_res_name, in_res_name, atom_table, cell, cell_std, wavelength, parameters):
    with open(in_res_name) as fo:
        res_lines = fo.readlines()
    atom_table = atom_table.copy()

    latt_line = [line.strip() for line in res_lines if line.upper().startswith('LATT ')][0]
    symm_lines = [line.strip() for line in res_lines if line.upper().startswith('SYMM ')]
    symm_string = '\n'.join(symm_lines)
    sfac_line = [line.strip() for line in res_lines if line.upper().startswith('SFAC ')][0]
    sfac_elements = [element.capitalize() for element in sfac_line.split()[1:]]
    unit_entries = ' '.join(['99'] * len(sfac_elements))
    sfac_df = pd.DataFrame({
        'sfac_index': np.arange(len(sfac_elements)) + 1,
        'type_symbol': sfac_elements
    })
    out_df = pd.merge(atom_table, sfac_df, on='type_symbol')
    atom_lines = '\n'.join([entries2atom_string(entries) for _, entries in out_df.iterrows()])

    output_res = f"""TITL har_out
CELL  {wavelength} {cell[0]:6.4f} {cell[1]:6.4f} {cell[2]:6.4f} {cell[3]:6.4f} {cell[4]:6.4f} {cell[5]:6.4f}
ZERR  999 {cell_std[0]:6.4f} {cell_std[1]:6.4f} {cell_std[2]:6.4f} {cell_std[3]:6.4f} {cell_std[4]:6.4f} {cell_std[5]:6.4f}
{latt_line}
{symm_string}
{sfac_line}
UNIT {unit_entries}
LIST 6
L.S. 0
FMAP 2
WGHT    0.000000
FVAR       {parameters[0]:8.6f}
{atom_lines}
HKLF 4
END
"""
    with open(out_res_name, 'w') as fo:
        fo.write(output_res)


def value_with_esd(values, esds):
    assert len(values.shape) == 1, 'Multidimensional array currently not supported'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        indexes = np.isfinite(1 / esds)
        orders = np.floor(np.log10(esds))
        smaller2 = np.full_like(values, False)
        smaller2[indexes] = np.array(esds[indexes]) * 10**(-orders[indexes]) < 2

        orders[np.logical_and(smaller2, orders)] -= 1
    strings = []
    for value, esd, order, index in zip(values, esds, orders, indexes):
        if index:
            format_dict = {'value': np.round(value, int(-order)),
                           'esd_val': int(np.round(esd / 10**(order)))}
            string = '{{value:0.{format_order}f}}({{esd_val}})'.format(format_order=int(-order))
            string = string.format(**format_dict)
            strings.append(string)
        else:
            string = f'{np.round(value, 20)}'
            strings.append(string)
    return strings


def write_incomp_cif(out_cif_name,
                     source_cif_name,
                     symm_strings, 
                     cell, 
                     cell_std, 
                     atom_table, 
                     parameters,
                     construction_instructions,
                     var_cov_mat,
                     options_dict,
                     refine_dict,
                     r_f,
                     r_f_strong,
                     wr2,
                     gof):
    source_cif = list(ciflike_to_dict(source_cif_name).values())[0]
    cell_mat_m = cell_constants_to_M(*cell)
    constr_xyz_esd, constr_uij_esd, *_ = construct_esds(var_cov_mat, construction_instructions)
    constructed_xyz, constructed_uij, *_ = construct_values(parameters, construction_instructions, cell_mat_m)

    symm_string = "'" + "'\n'".join(symm_strings) + "'"
    cell_strings = value_with_esd(cell, cell_std)

    xyz_strings = [value_with_esd(values, esds) for values, esds in zip(constructed_xyz.T, constr_xyz_esd.T)]
    xyz_zip = zip(atom_table['label'], atom_table['type_symbol'], *xyz_strings, atom_table['adp_type'])
    xyz_string = '\n'.join([' '.join(xyz_values) for xyz_values in xyz_zip])

    uij_strings = [value_with_esd(values, esds) for values, esds in zip(constructed_uij.T, constr_uij_esd.T)]
    uij_zip = zip(atom_table['label'], *uij_strings)
    uij_string = '\n'.join([' '.join(uij_values) for uij_values in uij_zip])

    source_bond_table = next(table for table in source_cif['loops'] if 'geom_bond_atom_site_label_1' in table.columns)
    bonds = [(line['geom_bond_atom_site_label_1'], line['geom_bond_atom_site_label_2']) for _, line in source_bond_table.iterrows() if line['geom_bond_site_symmetry_2'] == '.']
    distance_esd = [distance_with_esd(*bond, construction_instructions, parameters, var_cov_mat, cell, cell_std) for bond in bonds]
    distances, dist_esds = [np.array(value) for value in zip(*distance_esd)]
    distance_strings = value_with_esd(distances, dist_esds)
    distance_string = '\n'.join([' '.join((atom1, atom2, uij_string)) for (atom1, atom2), uij_string in zip(bonds, distance_strings)])

    out = f"""data_har
_space_group_crystal_system       {source_cif['space_group_crystal_system']}
_space_group_IT_number            {source_cif['space_group_IT_number']}
_space_group_name_H-M_alt         '{source_cif['space_group_name_H-M_alt']}'
_space_group_name_Hall            '{source_cif['space_group_name_Hall']}'

loop_
 _space_group_symop_operation_xyz
 {symm_string}
 
_cell_length_a                    {cell_strings[0]}
_cell_length_b                    {cell_strings[1]}
_cell_length_c                    {cell_strings[2]}
_cell_angle_alpha                 {cell_strings[3]}
_cell_angle_beta                  {cell_strings[4]}
_cell_angle_gamma                 {cell_strings[5]}

_refine_ls_R_factor_all           {r_f:0.4f}
_refine_ls_R_factor_gt            {r_f_strong:0.4f}
_refine_ls_wR_factor_ref          {wr2:0.4f}
_refine_ls_goodness_of_fit_ref    {gof:0.3f}

loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_adp_type
{xyz_string}

loop_
 _atom_site_aniso_label
 _atom_site_aniso_U_11
 _atom_site_aniso_U_22
 _atom_site_aniso_U_33
 _atom_site_aniso_U_23
 _atom_site_aniso_U_13
 _atom_site_aniso_U_12
{uij_string}

loop_
 _geom_bond_atom_site_label_1
 _geom_bond_atom_site_label_2
 _geom_bond_distance
{distance_string}

_refine_special_details
;
internal parameters:
xc: {options_dict['xc']}
h: {options_dict['h']}
core: {refine_dict['core']}
extinction: {refine_dict['extinction']}
grid_mult: {options_dict['gridrefinement']}
mode: {options_dict['mode']}
basis: {options_dict['basis']}
density_conv: {options_dict['convergence']['density']}
kpts: {options_dict['kpts']['size'][0]}
;
"""

    with open(out_cif_name, 'w') as fo:
        fo.write(out)