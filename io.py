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
                        elif test == '?':
                            datablocks[current_block][line_split[0][1:]] = None
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

    try:
        disp_corr_table = [table for table in cif['loops'] if 'atom_type_scat_dispersion_real' in table.columns][0].copy()
        disp_corr_table.columns = [label.replace('atom_', '') for label in disp_corr_table.columns]

        atom_table = pd.merge(atom_table, disp_corr_table, on='type_symbol', how='left') # add f' and f'' parameters
    except:
        warnings.warn('Could not find anomalous dispersion factors in cif file. You need to add them manually')

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
    try:
        wavelength = cif['diffrn_radiation_wavelength']
    except:
        warnings.warn('No wavelength found in cif file. You need to add it manually!')
        wavelength = None
    return atom_table, cell, cell_std, symm_mats_vecs, symm_strings, wavelength


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



def write_fcf(filename, hkl, refine_dict, parameters, symm_strings, construction_instructions, fjs, cell, dataset_name, fcf_mode):
    hkl = hkl.copy()
    cell_mat_m = cell_constants_to_M(*cell)
    constructed_xyz, constructed_uij, constructed_cijk, constructed_dijkl, constructed_occupancies = construct_values(parameters, construction_instructions, cell_mat_m)
    symm_list = [symm_to_matrix_vector(instruction) for instruction in symm_strings]
    symm_mats_r, symm_vecs_t = zip(*symm_list)
    symm_mats_vecs = (np.array(symm_mats_r), np.array(symm_vecs_t))
    cell_mat_f = np.linalg.inv(cell_mat_m).T
    index_vec_h = hkl[['h', 'k', 'l']].values

    wavelength = refine_dict['wavelength']

    intensity = hkl['intensity'].values.copy()
    stderr = hkl['stderr'].values.copy()

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

    if refine_dict['core'] == 'scale':
        extinction_parameter = 2
    else:
        extinction_parameter = 1

    if refine_dict['extinction'] == 'none':
        hkl['intensity'] = np.array(intensity / parameters[0])
        hkl['stderr'] = np.array(stderr / parameters[0])
    else:
        i_calc0 = np.abs(structure_factors)**2
        if refine_dict['extinction'] == 'secondary':
            # Secondary exctinction, as shelxl needs a wavelength                
            hkl['intensity'] = np.array(intensity / parameters[0] * (1 + parameters[extinction_parameter] * i_calc0))
            hkl['stderr'] = np.array(stderr / parameters[0] * (1 + parameters[extinction_parameter] * i_calc0))

        else:
            sintheta = np.linalg.norm(np.einsum('xy, zy -> zx', cell_mat_f, index_vec_h), axis=1) / 2 * wavelength
            sintwotheta = 2 * sintheta * np.sqrt(1 - sintheta**2)
            extinction_factors = 0.001 * wavelength**3 / sintwotheta
            hkl['intensity'] = np.array(intensity / parameters[0] * np.sqrt(1 + parameters[extinction_parameter] * extinction_factors * i_calc0))
            hkl['stderr'] = np.array(stderr / parameters[0] * np.sqrt(1 + parameters[extinction_parameter] * extinction_factors * i_calc0))


    if fcf_mode == 6:
        dispersion_real = jnp.array([atom.dispersion_real for atom in construction_instructions])
        dispersion_imag = jnp.array([atom.dispersion_imag for atom in construction_instructions])
        f_dash = dispersion_real + 1j * dispersion_imag

        #fjs_corr = np.zeros_like(fjs)
        #fjs_corr += f_dash[None,:,None]


        #structure_factors_corr = np.array(calc_f(
        #    xyz=constructed_xyz,
        #    uij=constructed_uij,
        #    cijk=constructed_cijk,
        #    dijkl=constructed_dijkl,
        #    occupancies=constructed_occupancies,
        #    index_vec_h=index_vec_h,
        #    cell_mat_f=cell_mat_f,
        #    symm_mats_vecs=symm_mats_vecs,
        #    fjs=fjs_corr
        #))

        #f_obs_sq = hkl['intensity'].values.copy()
        #f_obs_sq[f_obs_sq < 0] = 0
        #f_obs = np.sqrt(f_obs_sq)
        #f_obs = f_obs * np.exp(1j * np.angle(structure_factors))
        #hkl['intensity'] = np.abs(f_obs + structure_factors_corr)**2

        structure_factors = np.array(calc_f(
            xyz=constructed_xyz,
            uij=constructed_uij,
            cijk=constructed_cijk,
            dijkl=constructed_dijkl,
            occupancies=constructed_occupancies,
            index_vec_h=index_vec_h,
            cell_mat_f=cell_mat_f,
            symm_mats_vecs=symm_mats_vecs,
            fjs=fjs #- f_dash[None, :, None]
        ))

        hkl['abs(f_calc)'] = np.abs(structure_factors)
        hkl['phase_angle'] = np.rad2deg(np.angle(structure_factors)) % 360
        template = '{h:>4d}{k:>4d}{l:>4d} {intensity:13.2f} {stderr:13.2f} {abs(f_calc):13.2f} {phase_angle:7.1f}\n'
        #template = '{h} {k} {l} {intensity:.2f} {stderr:.2f} {abs(f_calc):.2f} {phase_angle:.1f}\n'
        columns = [
            'refln_index_h',
            'refln_index_k',
            'refln_index_l',
            'refln_F_squared_meas',
            'refln_F_squared_sigma',
            'refln_F_calc',
            'refln_phase_calc'
        ]

        start = '#\n# h,k,l, Fo-squared, sigma(Fo-squared), Fc and phi(calc)\n#'
    elif fcf_mode == 4:
        hkl['i_calc'] = np.abs(structure_factors)**2
        hkl['observed'] = 'o'
        template = '{h:>4d}{k:>4d}{l:>4d} {i_calc:13.2f} {intensity:13.2f} {stderr:13.2f} {observed}\n'
        columns = [
            'refln_index_h',
            'refln_index_k',
            'refln_index_l',
            'refln_F_squared_calc',
            'refln_F_squared_meas',
            'refln_F_squared_sigma',
            'refln_observed_status'
        ]
        start = '#\n# h,k,l, Fc-squared, Fo-squared, sigma(Fo-squared), status flag\n#'
    else:
        raise NotImplementedError(f'fcf mode {fcf_mode} is currently not implemented')


    hkl_out = hkl.copy()
    for index in ('h', 'k', 'l'):
        values = hkl_out[index].values.copy()
        values[values < 0] += 10000
        hkl_out[index + '_sort'] = values
    hkl_out['indexes'] = list(range(len(hkl_out)))
    hkl_out = hkl_out.sort_values(['l_sort', 'k_sort', 'h_sort'])
    hkl_out = hkl_out.astype({'h': np.int64, 'k': np.int64, 'l': np.int64 }).copy()

    lines = [''] * len(hkl_out)
    for index, line in hkl_out.iterrows():
        format_dict = {**line}
        format_dict['h'] = int(format_dict['h'])
        format_dict['k'] = int(format_dict['k'])
        format_dict['l'] = int(format_dict['l'])
        lines[index] = template.format(**format_dict)
        #print(format_dict)
        #print(line)
    lines_str = ''.join(lines)

    symm_string = "'" + "'\n'".join(symm_strings) + "'"

    loop2_string = '\nloop_\n _' + '\n _'.join(columns)

    output = [
        start,
        f'data_{dataset_name}\n',
        cif_entry_string('shelx_refln_list_code', fcf_mode),
        cif_entry_string('shelx_F_calc_maximum', float(np.round(np.max(np.abs(structure_factors)), 4))),
        f'\nloop_\n _space_group_symop_operation_xyz\n{symm_string}\n',
        cif_entry_string('cell_length_a', float(cell[0])),
        cif_entry_string('cell_length_b', float(cell[1])),
        cif_entry_string('cell_length_c', float(cell[2])),
        cif_entry_string('cell_angle_alpha', float(cell[3])),
        cif_entry_string('cell_angle_beta', float(cell[4])),
        cif_entry_string('cell_angle_gamma', float(cell[5])),
        loop2_string,
        lines_str,
        ''
    ]
    with open(filename, 'w') as fo:
        fo.write('\n'.join(output))

    return hkl


def entries2atom_string(label, sfac_index, xyz, uij, occupancy):
    strings = [
        label,
        str(sfac_index),
        '{:8.6f}'.format((xyz[0])),
        '{:8.6f}'.format((xyz[1])),
        '{:8.6f}'.format((xyz[2])),
        '{:8.5f}'.format((occupancy + 10)),
        '{:7.5f}'.format((uij[0])),
        '{:7.5f}'.format((uij[1])),
        '{:7.5f}'.format((uij[2])),
        '{:7.5f}'.format((uij[3])),
        '{:7.5f}'.format((uij[4])),
        '{:7.5f}'.format((uij[5]))
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


def write_res(out_res_name, in_res_name, cell, cell_std, wavelength, parameters, construction_instructions):
    with open(in_res_name) as fo:
        res_lines = fo.readlines()
    cell_mat_m = cell_constants_to_M(*cell)
    xyzs, uijs, _, _, occs = construct_values(parameters, construction_instructions, cell_mat_m)

    latt_line = [line.strip() for line in res_lines if line.upper().startswith('LATT ')][0]
    symm_lines = [line.strip() for line in res_lines if line.upper().startswith('SYMM ')]
    symm_string = '\n'.join(symm_lines)
    sfac_line = [line.strip() for line in res_lines if line.upper().startswith('SFAC ')][0]
    sfac_elements = [element.capitalize() for element in sfac_line.split()[1:]]
    unit_entries = ' '.join(['99'] * len(sfac_elements))
    sfacs = [sfac_elements.index(instr.element.capitalize()) + 1 for instr in construction_instructions]
    entry_zip = zip(construction_instructions, sfacs, xyzs, uijs, occs)
    atom_lines = '\n'.join([entries2atom_string(inst.name, sfac, xyz, uij, occ) for inst, sfac, xyz, uij, occ in entry_zip])

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
        raise NotImplementedError(f'{type(value)} is not implemented')
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


def create_atom_site_table_string(parameters, construction_instructions, cell, cell_std, var_cov_mat, crystal_system):
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
        uiso, uiso_esd = u_iso_with_esd(instr.name, construction_instructions, parameters, var_cov_mat, cell, cell_std, crystal_system)
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


def create_distance_table(bonds, construction_instructions, parameters, var_cov_mat, cell, cell_std, crystal_system):
    columns =  ['geom_bond_atom_site_label_1',
                'geom_bond_atom_site_label_2',
                'geom_bond_distance']
    string = '\nloop_\n _' + '\n _'.join(columns) + '\n'
    distances_esds = [distance_with_esd(bond[0], bond[1], construction_instructions, parameters, var_cov_mat, cell, cell_std, crystal_system) for bond in bonds]
    distances, distance_esds = zip(*distances_esds)
    distance_strings = value_with_esd(np.array(distances), np.array(distance_esds))
    string += ''.join([f'{atom1} {atom2} {distance_string}\n' for (atom1, atom2), distance_string in zip(bonds, distance_strings)])
    return string


def create_angle_table(angle_names, construction_instructions, parameters, var_cov_mat, cell, cell_std, crystal_system):
    columns =  ['geom_angle_atom_site_label_1',
                'geom_angle_atom_site_label_2',
                'geom_angle_atom_site_label_3',
                'geom_angle']
    string = '\nloop_\n _' + '\n _'.join(columns) + '\n'
    angles_esds = [angle_with_esd(angle_name[0], angle_name[1], angle_name[2], construction_instructions, parameters, var_cov_mat, cell, cell_std, crystal_system) for angle_name in angle_names]
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

def create_diff_density_entries(symm_mats_vecs, index_vec_h, scaled_intensity, structure_factors, cell_mat_m, dims=np.array([53, 53, 53])):
    symm_mats, symm_vecs = symm_mats_vecs
    deltafs2 = np.sqrt(scaled_intensity) - np.abs(structure_factors)
    xxx, yyy, zzz = np.meshgrid(*[np.linspace(0, 1, dim, endpoint=False) for dim in dims], indexing='ij')
    xyz = np.array([xxx, yyy, zzz])
    angles = np.angle(structure_factors)
    diff = np.zeros_like(xxx)
    for hkl_ind, deltaf, angle in zip(index_vec_h, deltafs2, angles):
        hkl_symm = np.einsum('x, axy -> ay', hkl_ind, symm_mats)
        hkl_symm, unique_indexes = np.unique(hkl_symm, axis=0, return_index=True)
        shifts = 2 * np.pi * np.einsum('x, ax -> a', hkl_ind, symm_vecs)[unique_indexes]
        
        diff += deltaf * np.sum(np.cos(2 * np.pi * np.einsum('ax, xijk -> aijk', hkl_symm, xyz) + angle - shifts[:, None, None, None]), axis=0)
    diff = diff / np.linalg.det(cell_mat_m)

    return '\n'.join([
        cif_entry_string('refine_diff_density_max', float(np.round(np.max(diff), 4))),
        cif_entry_string('refine_diff_density_min', float(np.round(np.min(diff), 4))),
        cif_entry_string('refine_diff_density_rms', float(np.round(np.std(diff), 4)))
    ])


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

    crystal_system = shelx_cif['space_group_crystal_system']

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

    refinement_string = """ - Structure optimisation was done using derivatives
   calculated with the python package JAX and
   BFGS minimisation in scipy.optimize.minimize"""

    if ishar:
        refinement_string += f"""
  - Refinement was done using structure factors
    derived from Hirshfeld densities
  - Density calculation was done with ASE/GPAW using the
    following settings
      xc: {options_dict['xc']}
      h: {options_dict['h']}
      core: {refine_dict['core']}
      grid_mult: {options_dict['gridrefinement']}
      density_conv: {options_dict['convergence']['density']}
      kpts: {options_dict['kpts']['size'][0]}"""
    else:
         refinement_string += f"""
 - Refinement was done using structure factors
   as usual for an IAM refinement"""
    if 'sphere' in source_cif['exptl_crystal_description']:
        crystal_dimension = add_from_cif('exptl_crystal_size_rad', source_cif)
    else:
        crystal_dimension = '\n'.join([
            add_from_cif('exptl_crystal_size_max', source_cif),
            add_from_cif('exptl_crystal_size_mid', source_cif),
            add_from_cif('exptl_crystal_size_min', source_cif)
        ])

    quality_dict = calculate_quality_indicators(construction_instructions, parameters, fjs, cell_mat_m, symm_mats_vecs, index_vec_h, intensity, stderr)

    bond_table = next(loop for loop in shelx_cif['loops'] if 'geom_bond_distance' in loop.columns)
    bonds = [(line['geom_bond_atom_site_label_1'],
              line['geom_bond_atom_site_label_2']) for _, line in bond_table.iterrows()
              if line['geom_bond_site_symmetry_2'] == '.']

    angle_table = next(loop for loop in shelx_cif['loops'] if 'geom_angle' in loop.columns)
    angle_names = [(line['geom_angle_atom_site_label_1'],
                    line['geom_angle_atom_site_label_2'],
                    line['geom_angle_atom_site_label_3']) for _, line in angle_table.iterrows() 
                    if line['geom_angle_site_symmetry_1'] == '.' and line['geom_angle_site_symmetry_3'] == '.']
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
        crystal_dimension,
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
        create_atom_site_table_string(parameters, construction_instructions, cell, cell_std, var_cov_mat, crystal_system),
        create_aniso_table_string(parameters, construction_instructions, cell, var_cov_mat),
        cif_entry_string('geom_special_details', """All esds are estimated using the full variance-covariance matrix.
Correlations between cell parameters are taken into account in the 
calculation of derivatives used for the error propagation to the esds
of U(iso), distances and angles. Otherwise the esds of the cell
parameters are assumed to be independent."""),
        create_distance_table(bonds, construction_instructions, parameters, var_cov_mat, cell, cell_std, crystal_system),
        create_angle_table(angle_names, construction_instructions, parameters, var_cov_mat, cell, cell_std, crystal_system),
        create_diff_density_entries(symm_mats_vecs, index_vec_h, intensity/parameters[0], structure_factors, cell_mat_m),
        create_fcf4_table(index_vec_h, structure_factors, intensity, stderr, parameters[0])
    ]
    with open(output_cif_name, 'w') as fo:
        fo.write('\n'.join(lines).replace('\n\n\n', '\n\n'))