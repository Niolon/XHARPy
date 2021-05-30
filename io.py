from collections import OrderedDict
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax
import re
import warnings
from .xharpy import (ConstrainedValues, cell_constants_to_M, distance_with_esd, construct_esds, construct_values, u_iso_with_esd, angle_with_esd,
                     calc_f)
from .quality import calculate_quality_indicators


def ciflike_to_dict(filename, return_descr=None, resolve_std=True):
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
                        elif (re.search(r'[^\d]', test) is None):
                            datablocks[current_block][line_split[0][1:]] = int(test)
                        elif re.search(r'[^\d^\.]', test) is None and re.search(r'\d', test) is not None:
                            datablocks[current_block][line_split[0][1:]] = float(test)
                        elif re.search(r'[\(\)]', test) is not None and re.search(r'[^\d^\.^\(^\)\-\+]', test) is None:
                            val, error = split_error(test)
                            datablocks[current_block][line_split[0][1:]] = val
                            datablocks[current_block][line_split[0][1:] + '_std'] = error
                        elif test.startswith('-'):
                            # This accounts for negative values without also catching dates
                            if (re.search(r'[^\d]', test[1:]) is None):
                                datablocks[current_block][line_split[0][1:]] = int(test)
                            elif re.search(r'[^\-^\d^\.]', test[1:]) is None and re.search(r'\d', test[1:]) is not None:
                                datablocks[current_block][line_split[0][1:]] = float(test)
                            else:
                                datablocks[current_block][line_split[0][1:]] = line_split[1]
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

    if return_descr is None:
        return datablocks
    elif type(return_descr) is int:
        return datablocks[list(datablocks.keys())[return_descr]]
    elif type(return_descr) is str:
        return datablocks[return_descr]
    else:
        raise ValueError('Invalid return_descr value. Must be either None, index as int or name as str')

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
    cif = ciflike_to_dict(cif_name, return_descr=cif_dataset)
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

    if all(atom_table['adp_type'] == 'Uiso'):
        atom_table[[
            'U_11', 'U_22', 'U_33', 'U_23', 'U_13', 'U_12', 'U_11_std', 'U_22_std',
            'U_33_std', 'U_23_std', 'U_13_std', 'U_12_std'
        ]] = np.nan
    else:
        adp_table = [table for table in cif['loops'] if 'atom_site_aniso_label' in table.columns][0].copy()
        adp_table.columns = [label.replace('atom_site_aniso_', '') for label in adp_table.columns]
        atom_table = pd.merge(atom_table, adp_table, on='label', how='left').copy() # put adp parameters into table


    disp_corr_table = [table for table in cif['loops'] if 'atom_type_scat_dispersion_real' in table.columns][0].copy()
    disp_corr_table.columns = [label.replace('atom_', '') for label in disp_corr_table.columns]

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
    return ConstrainedValues(variable_indexes=jnp.array(variable_indexes),
                             multiplicators=jnp.array(multiplicators),
                             added_value=jnp.array(added_value),
                             special_position=True) 



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
    try:
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
                string = f'{np.round(value, 6)}'
                strings.append(string)
        return strings
    except AttributeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not np.isfinite(1 / esds):
                return f'{np.round(values, 6)}'
            else:
                order = np.floor(np.log10(esds))
                if esds * 10**(-order) < 2:
                    order -= 1
                format_dict = {'value': np.round(values, int(-order)),
                            'esd_val': int(np.round(esds / 10**(order)))}
                string = '{{value:0.{format_order}f}}({{esd_val}})'.format(format_order=int(-order))
                string = string.format(**format_dict)
                return string
                
    


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
"""
    if all([value in options_dict for value in ('xc', 'h', 'core', 'extinction', 'gridrefinement', 'mode', 'basis', 'convergence', 'kpts')]):
        out += f"""
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


def cif_entry_string(name, value, string_sign=True):
    if value is None:
        entry_str = '?'
    elif type(value) is str and (len(value) > 45 or '\n' in value):
        entry_str = f'\n;\n{value}\n;'
    elif type(value) is str:
        if string_sign:
            entry_str = f"'{value}'"
        else:
            entry_str = value
    elif type(value) is float:
        entry_str = f'{value}'
    elif type(value) is int:
        entry_str = f'{value}'
    else:
        print(value, type(value))
    return f'_{name:<32s}  {entry_str}'


def add_from_cif(name, cif, std=False):
    if std:
        std_name = name + '_std'
        if std_name in cif:
            return cif_entry_string(name,
                             value_with_esd(cif[name],
                                            cif[std_name]),
                             False)
    try: 
        return cif_entry_string(name, cif[name])
    except KeyError:
        return cif_entry_string(name, None)


def cif2atom_type_table_string(cif, versionmajor, versionminor, ishar=True):
    table = next(loop for loop in cif['loops'] if 'atom_type_symbol' in loop.columns)
    if ishar:
        table['atom_type_scat_source'] = f'HAR in xHARPy {versionmajor}.{versionminor}'
    else:
        table['atom_type_scat_source'] = 'International Tables Vol C Tables 4.2.6.8 and 6.1.1.4'
    columns = [column for column in table.columns if not column.endswith('_std')]
    string = '\nloop_\n _' + '\n _'.join(columns) + '\n'
    template = (" '{atom_type_symbol}' '{atom_type_description}' "
     + '{atom_type_scat_dispersion_real:6.4f} {atom_type_scat_dispersion_imag:6.4f} '
     + "'{atom_type_scat_source}'\n")
    for index, row in table.iterrows():
        string += template.format(**row)
    return string


def cif2space_group_table_string(cif):
    table = next(loop for loop in cif['loops'] if 'space_group_symop_operation_xyz' in loop.columns)
    columns = [column for column in table.columns if not column.endswith('_std')]
    string = '\nloop_\n _' + '\n _'.join(columns) + '\n'
    template = " '{space_group_symop_operation_xyz}'\n"
    for _, row in table.iterrows():
        string += template.format(**row)
    return string


def create_atom_site_table_string(parameters, construction_instructions, cell, cell_std, var_cov_mat):
    columns = ['label', 'type_symbol', 'fract_x', 'fract_y', 'fract_z',
               'U_iso_or_equiv', 'adp_type', 'occupancy', 'site_symmetry_order']
    columns = ['atom_site_' + name for name in columns]
    string = '\nloop_\n _' + '\n _'.join(columns) + '\n'
    cell_mat_m = cell_constants_to_M(*cell)
    constructed_xyz, *_ = construct_values(parameters, construction_instructions, cell_mat_m)
    constr_xyz_esd, *_ = construct_esds(var_cov_mat, construction_instructions)

    for index, (xyz, xyz_esd, instr) in enumerate(zip(constructed_xyz, constr_xyz_esd, construction_instructions)):
        if type(instr.uij) is tuple:
            adp_type = 'Uani'
        elif type(instr.uij).__name__ == 'Uiso':
            adp_type = 'Uiso'
        elif type(instr.uij).__name__ == 'UEquivCalculated':
            adp_type = 'calc'
        else:
            raise NotImplementedError('There was a currently not implemented ADP calculation type')

        if instr.occupancy.special_position:
            occupancy = 1.0
            symmetry_order = int(1 / instr.occupancy.value)
        else:
            occupancy = instr.occupancy.value
            symmetry_order = 1

        position_string = ' '.join(value_with_esd(xyz, xyz_esd))
        uiso, uiso_esd = u_iso_with_esd(instr.name, construction_instructions, parameters, var_cov_mat, cell, cell_std)
        uiso_string = value_with_esd(float(uiso), float(uiso_esd))
        string += f'{instr.name} {instr.element} {position_string} {uiso_string} {adp_type} {occupancy} {symmetry_order}\n'
    return string


def create_aniso_table_string(parameters, construction_instructions, cell, var_cov_mat):
    columns = ['label', 'U_11', 'U_22', 'U_33', 'U_23', 'U_13', 'U_12']
    columns = ['atom_site_aniso_' + name for name in columns]
    string = '\nloop_\n _' + '\n _'.join(columns) + '\n'

    cell_mat_m = cell_constants_to_M(*cell)
    _, uijs, *_ = construct_values(parameters, construction_instructions, cell_mat_m)
    _, uij_esds, *_ = construct_esds(var_cov_mat, construction_instructions)

    for instr, uij, uij_esd in zip(construction_instructions, uijs, uij_esds):
        if type(instr.uij) is tuple:
            # we have an anisotropic adp
            uij_string = ' '.join(value_with_esd(uij, uij_esd))
            string += f'{instr.name} {uij_string}\n'
    return string


def create_distance_table(bonds, construction_instructions, parameters, var_cov_mat, cell, cell_std):
    columns =  ['geom_bond_atom_site_label_1',
                'geom_bond_atom_site_label_2',
                'geom_bond_distance']
    string = '\nloop_\n _' + '\n _'.join(columns) + '\n'
    distances_esds = [distance_with_esd(bond[0], bond[1], construction_instructions, parameters, var_cov_mat, cell, cell_std) for bond in bonds]
    distances, distance_esds = zip(*distances_esds)
    distance_strings = value_with_esd(np.array(distances), np.array(distance_esds))
    string += ''.join([f'{atom1} {atom2} {distance_string}\n' for (atom1, atom2), distance_string in zip(bonds, distance_strings)])
    return string


def create_angle_table(angle_names, construction_instructions, parameters, var_cov_mat, cell, cell_std):
    columns =  ['geom_angle_atom_site_label_1',
                'geom_angle_atom_site_label_2',
                'geom_angle_atom_site_label_3',
                'geom_angle']
    string = '\nloop_\n _' + '\n _'.join(columns) + '\n'
    angles_esds = [angle_with_esd(angle_name[0], angle_name[1], angle_name[2], construction_instructions, parameters, var_cov_mat, cell, cell_std) for angle_name in angle_names]
    angles, angle_esds = zip(*angles_esds)
    angle_strings = value_with_esd(np.array(angles), np.array(angle_esds))
    string += ''.join([f'{atom1} {atom2} {atom3} {angle_string}\n' for (atom1, atom2, atom3), angle_string in zip(angle_names, angle_strings)])
    return string


def create_fcf4_table(index_vec_h, structure_factors, intensity, stderr, scaling):
    columns =  ['refln_index_h',
                'refln_index_k',
                'refln_index_l',
                'refln_F_squared_calc',
                'refln_F_squared_meas',
                'refln_F_squared_sigma']
    string = '\nloop_\n _' + '\n _'.join(columns) + '\n'

    for (h, k, l), i_calc, i_meas, esd_meas in zip(index_vec_h, np.abs(structure_factors)**2, intensity / scaling, stderr / scaling):
        string += f'{h:>4d}{k:>4d}{l:>4d}{i_calc:14.2f} {i_meas:14.2f}{esd_meas:14.2f}\n'
    return string

def write_cif(output_cif_name,
              dataset_name,
              shelx_cif_name,
              shelx_descr,
              source_cif_name,
              source_descr,
              fjs,
              parameters,
              var_cov_mat,
              construction_instructions,
              symm_mats_vecs,
              hkl,
              shift_ov_su,
              options_dict,
              refine_dict,
              cell,
              cell_std):
    versionmajor = 0
    versionminor = 1

    shelx_cif = ciflike_to_dict(shelx_cif_name, shelx_descr)
    source_cif = ciflike_to_dict(source_cif_name, source_descr)

    hkl['strong_condition'] = hkl['intensity'] / hkl['stderr'] > 2
    index_vec_h = hkl[['h', 'k', 'l']].values
    intensity = hkl['intensity'].values
    stderr = hkl['stderr'].values
    cell_mat_m = cell_constants_to_M(*cell)
    cell_mat_f = np.linalg.inv(cell_mat_m).T
    ishar = all([value in options_dict for value in ('xc', 'h', 'core', 'extinction', 'gridrefinement', 'mode', 'basis', 'convergence', 'kpts')])
    constructed_xyz, constructed_uij, constructed_cijk, constructed_dijkl, constructed_occupancies = construct_values(parameters, construction_instructions, cell_mat_m)

    structure_factors = np.array(calc_f(
        xyz=constructed_xyz,
        uij=constructed_uij,
        cijk=constructed_cijk,
        dijkl=constructed_dijkl,
        occupancies=constructed_occupancies,
        index_vec_h=index_vec_h,
        cell_mat_f=cell_mat_f,
        symm_mats_vecs=symm_mats_vecs,
        fjs=fjs
    ))

    refinement_string = """ . Structure optimisation was done using derivatives
   calculated with the python package JAX and
   BFGS minimisation in scipy.optimize.minimize"""

    if ishar:
        refinement_string += f"""
  . Refinement was done using structure factors
    derived from Hirshfeld densities
  . Density calculation was done with ASE/GPAW using the
    following settings
      xc: {options_dict['xc']}
      h: {options_dict['h']}
      core: {refine_dict['core']}
      grid_mult: {options_dict['gridrefinement']}
      density_conv: {options_dict['convergence']['density']}
      kpts: {options_dict['kpts']['size'][0]}"""
    else:
         refinement_string += f"""
 . Refinement was done using structure factors
   as usual for an IAM refinement"""

    quality_dict = calculate_quality_indicators(construction_instructions, parameters, fjs, cell_mat_m, symm_mats_vecs, index_vec_h, intensity, stderr)

    bond_table = next(loop for loop in shelx_cif['loops'] if 'geom_bond_distance' in loop.columns)
    bonds = [(line['geom_bond_atom_site_label_1'],
              line['geom_bond_atom_site_label_2']) for _, line in bond_table.iterrows()]

    angle_table = next(loop for loop in shelx_cif['loops'] if 'geom_angle' in loop.columns)
    angle_names = [(line['geom_angle_atom_site_label_1'],
                    line['geom_angle_atom_site_label_2'],
                    line['geom_angle_atom_site_label_3']) for _, line in angle_table.iterrows()]
    lines = [
        f'\ndata_{dataset_name}\n',
        cif_entry_string('audit_creation_method', f'xHARPY {versionmajor}.{versionminor}'),
        add_from_cif('chemical_name_systematic', source_cif),
        add_from_cif('chemical_name_common', source_cif),
        add_from_cif('chemical_melting_point', source_cif),
        add_from_cif('chemical_formula_moiety', source_cif),
        add_from_cif('chemical_formula_sum', source_cif),
        add_from_cif('chemical_formula_weight', source_cif),
        cif2atom_type_table_string(shelx_cif, 0, 1, ishar),
        add_from_cif('space_group_crystal_system', shelx_cif),
        add_from_cif('space_group_IT_number', shelx_cif),
        add_from_cif('space_group_name_H-M_alt', shelx_cif),
        add_from_cif('space_group_name_Hall', shelx_cif),
        cif2space_group_table_string(shelx_cif),
        add_from_cif('cell_length_a', shelx_cif, std=True),
        add_from_cif('cell_length_b', shelx_cif, std=True),
        add_from_cif('cell_length_c', shelx_cif, std=True),
        add_from_cif('cell_angle_alpha', shelx_cif, std=True),
        add_from_cif('cell_angle_beta', shelx_cif, std=True),
        add_from_cif('cell_angle_gamma', shelx_cif, std=True),
        add_from_cif('cell_volume', shelx_cif, std=True),
        add_from_cif('cell_formula_units_Z', shelx_cif),
        add_from_cif('cell_measurement_temperature', source_cif, std=True),
        add_from_cif('cell_measurement_reflns_used', source_cif),
        add_from_cif('cell_measurement_theta_min', source_cif),
        add_from_cif('cell_measurement_theta_max', source_cif),
        '',
        add_from_cif('exptl_crystal_description', source_cif),
        add_from_cif('exptl_crystal_colour', source_cif),
        add_from_cif('exptl_crystal_density_meas', source_cif),
        add_from_cif('exptl_crystal_density_method', source_cif),
        add_from_cif('exptl_crystal_density_diffrn', shelx_cif),
        add_from_cif('exptl_crystal_F_000', shelx_cif),
        add_from_cif('exptl_transmission_factor_min', source_cif),
        add_from_cif('exptl_transmission_factor_max', source_cif),
        add_from_cif('exptl_crystal_size_max', source_cif),
        add_from_cif('exptl_crystal_size_mid', source_cif),
        add_from_cif('exptl_crystal_size_min', source_cif),
        add_from_cif('exptl_absorpt_coefficient_mu', shelx_cif),
        add_from_cif('exptl_absorpt_correction_type', source_cif),
        add_from_cif('exptl_absorpt_correction_T_min', source_cif),
        add_from_cif('exptl_absorpt_correction_T_max', source_cif),
        add_from_cif('exptl_absorpt_process_details', source_cif),
        add_from_cif('exptl_absorpt_special_details', source_cif),
        '',
        add_from_cif('diffrn_ambient_temperature', source_cif, std=True),
        add_from_cif('diffrn_radiation_wavelength', source_cif),
        add_from_cif('diffrn_radiation_type', source_cif),
        add_from_cif('diffrn_source', source_cif),
        add_from_cif('diffrn_measurement_device_type', source_cif),
        add_from_cif('diffrn_measurement_method', source_cif),
        add_from_cif('diffrn_detector_area_resol_mean', source_cif),
        add_from_cif('diffrn_reflns_number', source_cif),
        add_from_cif('diffrn_reflns_av_unetI/netI', source_cif),
        add_from_cif('diffrn_reflns_av_R_equivalents', source_cif),
        add_from_cif('diffrn_reflns_limit_h_min', source_cif),
        add_from_cif('diffrn_reflns_limit_h_max', source_cif),
        add_from_cif('diffrn_reflns_limit_k_min', source_cif),
        add_from_cif('diffrn_reflns_limit_k_max', source_cif),
        add_from_cif('diffrn_reflns_limit_l_min', source_cif),
        add_from_cif('diffrn_reflns_limit_l_max', source_cif),
        add_from_cif('diffrn_reflns_theta_min', source_cif),
        add_from_cif('diffrn_reflns_theta_max', source_cif),
        add_from_cif('diffrn_reflns_theta_full', source_cif),
        add_from_cif('diffrn_measured_fraction_theta_max', shelx_cif),
        add_from_cif('diffrn_measured_fraction_theta_full', shelx_cif),
        add_from_cif('diffrn_reflns_Laue_measured_fraction_max', shelx_cif),
        add_from_cif('diffrn_reflns_Laue_measured_fraction_full', shelx_cif),
        add_from_cif('diffrn_reflns_point_group_measured_fraction_max', shelx_cif),
        add_from_cif('diffrn_reflns_point_group_measured_fraction_full', shelx_cif),
        '',
        cif_entry_string('reflns_number_total', len(hkl)),
        cif_entry_string('reflns_number_gt', int(np.sum(hkl['strong_condition']))),
        cif_entry_string('reflns_threshold_expression', 'I > 2\s(I)'),
        add_from_cif('reflns_Friedel_coverage', shelx_cif),
        add_from_cif('reflns_Friedel_fraction_max', shelx_cif),
        add_from_cif('reflns_Friedel_fraction_full', shelx_cif),
        cif_entry_string(
            'reflns_special_details', 
            """ _reflns_Friedel_fraction is defined as the number of unique
Friedel pairs measured divided by the number that would be
possible theoretically, ignoring centric projections and
systematic absences."""
        ),
        '',
        add_from_cif('computing_data_collection', source_cif),
        add_from_cif('computing_cell_refinement', source_cif),
        add_from_cif('computing_data_reduction', source_cif),
        add_from_cif('computing_structure_solution', source_cif),
        cif_entry_string('computing_structure_refinement', f'xHARPY {versionmajor}.{versionminor}'),
        cif_entry_string('computing_molecular_graphics', None),
        cif_entry_string('computing_publication_material', f'xHARPY {versionmajor}.{versionminor}'),
        '',
        cif_entry_string('atom_sites_solution_hydrogens', 'difmap', False),
        '',
        cif_entry_string('refine_special_details', refinement_string),
        cif_entry_string('refine_ls_structure_factor_coef', 'Fsqd', False),
        cif_entry_string('refine_ls_matrix_type', 'full'), # TODO is full?
        cif_entry_string('refine_ls_weighting_scheme', 'sigma', False),
        cif_entry_string('refine_ls_weighting_details', 'w=1/[\s^2^(Fo^2^)]'),
        cif_entry_string('refine_ls_hydrogen_treatment', 'refall', False),
        cif_entry_string('refine_ls_extinction_method', 'none', False),
        cif_entry_string('refine_ls_extinction_coef', '.', False),
        cif_entry_string('refine_ls_number_reflns', len(hkl)),
        cif_entry_string('refine_ls_number_parameters', len(parameters)),
        cif_entry_string('refine_ls_number_restraints', 0),
        cif_entry_string('refine_ls_R_factor_all', float(np.round(quality_dict['R(F)'], 4))),
        cif_entry_string('refine_ls_R_factor_gt', float(np.round(quality_dict['R(F)(I>2s)'], 4))),
        cif_entry_string('refine_ls_wR_factor_ref', float(np.round(quality_dict['wR(F^2)'], 4))),
        cif_entry_string('refine_ls_wR_factor_gt', float(np.round(quality_dict['wR(F^2)(I>2s)'], 4))),
        cif_entry_string('refine_ls_goodness_of_fit_ref', float(np.round(quality_dict['GOF'], 3))),
        cif_entry_string('refine_ls_shift/su_max', float(np.round(np.max(shift_ov_su[0]), 3))),
        cif_entry_string('refine_ls_shift/su_mean', float(np.round(np.mean(shift_ov_su[0]), 3))),
        create_atom_site_table_string(parameters, construction_instructions, cell, cell_std, var_cov_mat),
        create_aniso_table_string(parameters, construction_instructions, cell, var_cov_mat),
        create_distance_table(bonds, construction_instructions, parameters, var_cov_mat, cell, cell_std),
        create_angle_table(angle_names, construction_instructions, parameters, var_cov_mat, cell, cell_std),
        create_fcf4_table(index_vec_h, structure_factors, intensity, stderr, parameters[0])
    ]
    with open(output_cif_name, 'w') as fo:
        fo.write('\n'.join(lines).replace('\n\n\n', '\n\n'))