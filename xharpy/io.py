"""This module containts the input/output routines for the XHARPy library"""

from collections import OrderedDict

import pandas as pd
import numpy as np
from typing import Any, Dict, Union, Tuple, List
from .common_jax import jnp
import re
import warnings
from io import StringIO
import pickle
import textwrap
from .defaults import (
    get_parameter_index, get_value_or_default, XHARPY_VERSION
)
from .conversion import cell_constants_to_M
try:
    from .refine import calc_f
except:
    warnings.warn('Refine module could not be imported, jax is probably missing')

from .structure.common import (
    AtomInstructions, FixedValue, RefinedValue
)
from .structure.initialise import (
    create_construction_instructions, ConstrainedValues
)
from .structure.construct import (
    distance_with_esd, construct_esds, construct_values, u_iso_with_esd,
    angle_with_esd
)
try:
    from .quality import calculate_quality_indicators
except:
    warnings.warn('quality module could not be imported, jax is probably missing')

from .conversion import calc_sin_theta_ov_lambda, cell_constants_to_M

def ciflike_to_dict(
    filename: str,
    return_descr: Union[str, int, None] = None,
    resolve_esd: bool =True
) -> Union[Dict[str, Dict], Dict]:
    """Function to read in cif or cif-like (e.g. fcf) files. Can return all 
    structures contained in the file. To return only the data of one structure
    use return_descr to select by dataset name or index.

    Parameters
    ----------
    filename : str
        Path to a cif or cif-like file
    return_descr : Union[str, int, None], optional
        Can be used to only return a specific dataset from the cif file
        if a string is given as argument, the dataset with that name is 
        returned. An integer will return the dataset by index (i.e. 0 will give 
        the first dataset in the file), None will return a dict with all
        datasets, with the dataset name as key, by default None
    resolve_esd : bool, optional
        If this argument is set to true, will split arguments, which have an
        esd into two arguments: arg, arg_esd, False will return a string in this
        case (i.e. '12(3)'), by default True

    Returns
    -------
    cif_content: Union[Dict[str, Dict], Dict]
        Returns a dictionary of dataset_name, dataset_dict pairs or a single 
        dataset as OrderedDict. Within the dataset all entries are given as
        key, value pairs with the key being the entry in the cif_file without
        the preceding underscore. The routine will try to cast the value into
        a float, int or string, depending on whether digits, digits and a dot
        or other characters are present. All loops are given as a list of pandas
        DataFrame objects under the 'loops' keyword. 

    Raises
    ------
    e
        If an exception occurs it is raised after printing the line in the cif 
        file for debugging purposes
    ValueError
        The return_descr was in an invalid type
    """
    PATTERN = re.compile(r'''((?:[^ "']|"[^"]*"|'[^']*')+)''')
    with open(filename, 'r') as fo:
        lines = [line for line in fo.readlines()]
    datablocks = OrderedDict()
    current_loop_lines = []
    current_loop_titles = []
    # If there is data before the first data entry store it as preblock
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
                    if resolve_esd:
                        for column in new_df.columns:
                            if new_df[column].dtype != 'O':
                                continue
                            concatenate = ''.join(new_df[column])
                            if  re.search(r'[\(\)]', concatenate) is not None and re.search(r'[^\d^\.^\(^\)\-\+]', concatenate) is None:
                                values, errors = np.array([split_error(val) for val in new_df[column]]).T
                                new_df[column] = values
                                new_df[column+'_esd'] = errors
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
                    if resolve_esd:
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
                            datablocks[current_block][line_split[0][1:] + '_esd'] = error
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
            if resolve_esd:
                for column in new_df.columns:
                    if new_df[column].dtype != 'O':
                        continue
                    concatenate = ''.join(new_df[column])
                    if  re.search(r'[\(\)]', concatenate) is not None and re.search(r'[^\d^\.^\(^\)\-\+]', concatenate) is None:
                        values, errors = np.array([split_error(val) for val in new_df[column]]).T
                        new_df[column] = values
                        new_df[column+'_esd'] = errors
            datablocks[current_block]['loops'].append(new_df)

    if return_descr is None:
        return datablocks
    elif type(return_descr) is int:
        return datablocks[list(datablocks.keys())[return_descr]]
    elif type(return_descr) is str:
        return datablocks[return_descr]
    else:
        raise ValueError('Invalid return_descr value. Must be either None, index as int or name as str')

def split_error(string: str) -> Union[Tuple[float, float], Tuple[int, int]]:
    """Helper function to split a string containing a value with error in
    brackets to a value-esd pair

    Parameters
    ----------
    string : str
        Input string containing the value to be split

    Returns
    -------
    Union[Tuple[float, float], Tuple[int, int]]
        Pair of floats if a '.' was present in string, otherwise a pair of ints
        containing the value and its esd
    """    
    int_search = re.search(r'([\-\d]*)\((\d*)\)', string)
    search = re.search(r'(\-{0,1})([\d]*)\.(\d*)\((\d*)\)', string)
    if search is not None:
        # we have found a float
        sign, before_dot, after_dot, err = search.groups()
        if sign == '-':
            return -1 * (int(before_dot) + int(after_dot) * 10**(-len(after_dot))), int(err) * 10**(-len(after_dot))
        else:
            return int(before_dot) + int(after_dot) * 10**(-len(after_dot)), int(err) * 10**(-len(after_dot))
    elif int_search is not None:
        # we have found an int
        value, error = int_search.groups()
        return int(value), int(error)
    else:
        # no error found
        return float(string), 0.0  


def symm_to_matrix_vector(instruction: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Converts a symmetry instruction into a symmetry matrix and a translation
    vector for that symmetry element.

    Parameters
    ----------
    instruction : str
        Instruction string containing symmetry instruction for all three 
        coordinates separated by comma signs (e.g -x, -y, 0.5+z)

    Returns
    -------
    symm_matrix: jnp.ndarray, 
        size (3, 3) array containing the symmetry matrix for the symmetry element
    symm_vector: jnp.ndarray
        size (3) array containing the translation vector for the symmetry element
    """    
    instruction_strings = [val.replace(' ', '').upper() for val in instruction.split(',')]
    matrix = np.zeros((3,3), dtype=np.float64)
    vector = np.zeros(3, dtype=np.float64)
    for xyz, element in enumerate(instruction_strings):
        # search for fraction in a/b notation
        fraction1 = re.search(r'(-{0,1}\d{1,3})/(\d{1,3})(?![XYZ])', element)
        # search for fraction in 0.0 notation
        fraction2 = re.search(r'(-{0,1}\d{0,1}\.\d{1,4})(?![XYZ])', element)
        # search for whole numbers
        fraction3 = re.search(r'(-{0,1}\d)(?![XYZ])', element)
        if fraction1:
            vector[xyz] = float(fraction1.group(1)) / float(fraction1.group(2))
        elif fraction2:
            vector[xyz] = float(fraction2.group(1))
        elif fraction3:
            vector[xyz] = float(fraction3.group(1))

        symm = re.findall(r'-{0,1}[\d\.]{0,8}[XYZ]', element)
        for xyz_match in symm:
            if len(xyz_match) == 1:
                sign = 1
            elif xyz_match[0] == '-' and len(xyz_match) == 2:
                sign = -1
            else:
                sign = float(xyz_match[:-1])
            if xyz_match[-1] == 'X':
                matrix[xyz, 0] = sign
            if xyz_match[-1] == 'Y':
                matrix[xyz, 1] = sign
            if xyz_match[-1] == 'Z':
                matrix[xyz, 2] = sign
    return jnp.array(matrix), jnp.array(vector)


def cif2data(
    cif_path: str,
    cif_dataset: Union[str, int] = 0
) -> Tuple[
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        Tuple[jnp.ndarray, jnp.ndarray],
        List[str],
        float
    ]:
    """Function to generate the needed variables used for the futher refinement
    from a given cif file.

    Parameters
    ----------
    cif_path : str
        Path to the cif file, that should be used as source for the structure 
        information
    cif_dataset : Union[str, int], optional
        dataset within that cif file, that is supposed to be used, when a string
        is passed as the argument, the dataset is selected by name. When an 
        integer is passed it is selected by index. If you have named with
        numbering, pass the number as a string (e.g. '2'), by default 0

    Returns
    -------
    atom_table: pd.DataFrame
        pandas DataFrame that contains the atomic information. Columns are named
        like their counterparts in the cif file but without the common start for
        each table (e.g. atom_site_fract_x -> fract_x). All tables are merged on
        the atom label. f' and f'' are merged on atom_type
    cell: np.ndarray
        array with the lattice constants (Angstroem, Degree)
    cell_esd: np.ndarray
        array with the estimated standard deviation of the lattice constants
        (Angstroem, Degree)
    symm_mats_vecs: Tuple[jnp.ndarray, jnp.ndarray]
        size (K, 3, 3) array of symmetry matrices and (K, 3) array of translation
        vectors for all symmetry elements in the unit cell
    symm_instructions: List[str]
        List of symmetry instructions from the cif-file. Needed for writing a
        new cif file
    wavelength: float
        Measurement wavelength in Angstroem
    """    
    cif = ciflike_to_dict(cif_path, return_descr=cif_dataset)
    cell = np.array([cif['cell_length_a'],
                     cif['cell_length_b'],
                     cif['cell_length_c'],
                     cif['cell_angle_alpha'],
                     cif['cell_angle_beta'],
                     cif['cell_angle_gamma']])

    std_keys = ['cell_length_a_esd', 'cell_length_b_esd', 'cell_length_c_esd',
                'cell_angle_alpha_esd', 'cell_angle_beta_esd', 'cell_angle_gamma_esd']

    cell_esd = np.array([cif[key] if key in list(cif.keys()) else 0.0 for key in std_keys])

    atom_table = [table for table in cif['loops'] if 'atom_site_label' in table.columns][0].copy()
    atom_table.columns = [label.replace('atom_site_', '') for label in atom_table.columns]
    if 'type_symbol' not in atom_table:
        atom_table['type_symbol'] = [str(re.match(r'([A-Za-z]{1,2})\d*', line['label']).groups(1)[0]) for _, line in atom_table.iterrows()]
    if 'site_symmetry_order' in atom_table:
        atom_table['occupancy'] /= atom_table['site_symmetry_order']
    atom_table = atom_table.rename({'thermal_displace_type': 'adp_type'}, axis=1).copy()

    if all(atom_table['adp_type'] == 'Uiso'):
        atom_table[[
            'U_11', 'U_22', 'U_33', 'U_23', 'U_13', 'U_12', 'U_11_esd', 'U_22_esd',
            'U_33_esd', 'U_23_esd', 'U_13_esd', 'U_12_esd'
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

    try:
        wavelength = cif['diffrn_radiation_wavelength']
    except:
        warnings.warn('No wavelength found in cif file. You need to add it manually!')
        wavelength = None
    return atom_table, cell, cell_esd, symm_mats_vecs, symm_strings, wavelength


def instructions_to_constraints(
    names: List[str],
    instructions: str
)-> ConstrainedValues:
    """Helper function to generate symmetry constraints for a special position
    from a single atom entry in the the shelxl .lst file

    Parameters
    ----------
    names : List[str]
        names of variables to search for at the call
    instructions : str
        instructions to search in

    Returns
    -------
    ConstrainedValues
        A valid ConstrainedValues instance for the set of variables
    """
    variable_indexes = list(range(len(names)))
    multiplicators = [1.0] * len(names)
    added_values = [0.0] * len(names)
    for index, name in enumerate(names):
        if name not in instructions:
            continue
        for following_index in range(index + 1, len(names)):
            variable_indexes[following_index] -= 1
        mult, var, add = instructions[name]
        if var == '':
            variable_indexes[index] = -1
            multiplicators[index] = 0.0
            added_values[index] = add
        else:
            variable_indexes[index] = variable_indexes[names.index(var)]
            multiplicators[index] = mult
            added_values[index] = add
    return ConstrainedValues(variable_indexes=jnp.array(variable_indexes),
                             multiplicators=jnp.array(multiplicators),
                             added_values=jnp.array(added_values),
                             special_position=True) 



def lst2constraint_dict(filename: str) -> Dict[str, Dict[str, ConstrainedValues]]:
    """Helper function to create a constraint dict for the refinement if atoms 
    on special positions are present. Make sure that atoms on special positions
    are actually refined in SHELXL as you plan to refine them  in the refinement
    in XHARPy. Use AFIX 1 if hydrogen atoms are located on special positions,
    but you want to refine them anisotropically in the Hirshfeld refinement or 
    add the ConstraintValues variables manually. As SHELXL cannot refine Gram-
    Charlier parameters, these need to be added manually for the time being.

    Parameters
    ----------
    filename : str
        Path to the shelxl .lst file

    Returns
    -------
    constraint_dict: Dict[str, Dict[str, ConstrainedValues]]
        Dictionary containing the generated constraints for the individual atoms
        and parameters
    """    
    with open(filename) as fo:
        lst_content = fo.read()

    find = re.search(r'Special position constraints.*?\n\n\n', lst_content, flags=re.DOTALL)  
    if find is None:
        return {}

    xyz_names = ['x', 'y', 'z']
    uij_names = ['U11', 'U22', 'U33', 'U23', 'U13', 'U12']
    occ_names = ['sof']
    names = xyz_names + uij_names + occ_names

    replace = {
        '0.3333': 1/3,
        '0.33333': 1/3,
        '0.6667': 2/3,
        '0.66667': 2/3,
        '0.1667': 1/6,
        '0.16667': 1/6,
        '0.8333': 5/6,
        '0.83333': 5/6
    }

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
                                mult = replace.get(prod_part, float(prod_part))
                    else:
                        if sum_part in names:
                            var = sum_part
                            mult = 1
                        add = replace.get(sum_part, float(sum_part))
                instructions[target] = (mult, var, add)
        constraint_dict[name] = {
            'xyz': instructions_to_constraints(xyz_names, instructions),
            'uij': instructions_to_constraints(uij_names, instructions),
            'occ': instructions_to_constraints(occ_names, instructions)
        }
    return constraint_dict


def shelxl_hkl2pd(hkl_name: str) -> pd.DataFrame:
    """Helper function to read in a shelx-style .hkl file as dataframe. Note
    that XHARPy does expect a merged hkl file, using an unmerged file will 
    be very slow and lead to unexpected esd values.

    Parameters
    ----------
    hkl_name : str
        Path to hkl file

    Returns
    -------
    hkl: pd.DataFrame
        Dataframe with named columns: 'h', 'k', 'l', 'intensity', 'esd_int' and
        possibly 'batch_no' if six columns are present in the file.

    Raises
    ------
    ValueError
        Number of columns is unexpected for a SHELX hkl file
    """    
    with open(hkl_name, 'r') as fo:
        content = fo.read()

    # if zero line in there use as end
    content = content.split('   0   0   0    0.00    0.00')[0]
    df = pd.read_fwf(StringIO(content), widths=(4,4,4,8,8))
    df.columns = ['h', 'k', 'l', 'intensity', 'esd_int']
    return df

def xd_hkl2pd(
    path: str
) -> pd.DataFrame:
    """Returns the reflection intensity DataFrame from an XD hkl file.
    Currently, only F^2 with NDAT 7 is implemented as format

    Parameters
    ----------
    path : str
        Path to hkl file

    Returns
    -------
    hkl: pd.DataFrame
        Dataframe with named columns: 'h', 'k', 'l', 'dataset', 'intensity',
        'esd_int', 'scale'

    Raises
    ------
    NotImplementedError
        Format of xd.hkl currently not implemented.
    """
    with open(path, 'r') as fo:
        first_line = fo.readlines()[0]
    options = first_line.split()
    if options[1] != 'F^2' or options[3] != '7':
        raise NotImplementedError('Currently only F^2 and NDAT 7 is implemented')

    hkl = pd.read_csv(path, skiprows=1, header=None, sep='\s+')
    hkl.columns = ['h', 'k', 'l', 'dataset', 'intensity', 'esd_int', 'scale']
    return hkl

def fcf2hkl_pd(
    fcf_path: str,
    fcf_dataset: Union[str, int] = 0
) -> pd.DataFrame:
    """Helper function to generate a DataFrame from a given .fcf file. Might
    also work with cif files if a fcf-like loop is present

    Parameters
    ----------
    fcf_path : str
        Path to the fcf file
    fcf_dataset : Union[str, int], optional
        dataset within that fcf file, that is supposed to be used, when a string
        is passed as the argument, the dataset is selected by name. When an 
        integer is passed it is selected by index. If you have named with
        numbering, pass the number as a string (e.g. '2'), by default 0
    Returns
    -------
    hkl: pd.DataFrame
        Dataframe with named columns: 'h', 'k', 'l', 'intensity', 'esd_int'

    Raises
    ------
    ValueError
        The file contains no table with refln_F_squared_meas as a column
    """
    fcf = ciflike_to_dict(fcf_path, fcf_dataset)
    try:
        table = next(loop for loop in fcf['loops'] if 'refln_F_squared_meas' in loop.columns)
    except StopIteration:
        raise ValueError("I could not find a table containing a 'refln_F_squared_meas' column in the given file")
    hkl = table[['refln_index_h', 'refln_index_k', 'refln_index_l', 'refln_F_squared_meas', 'refln_F_squared_sigma']].copy()
    hkl.columns = ['h', 'k', 'l', 'intensity', 'esd_int']
    return hkl


def write_fcf(
    fcf_path: str,
    fcf_dataset: str,
    fcf_mode: int,
    cell: np.ndarray,
    hkl: pd.DataFrame,
    construction_instructions: List[AtomInstructions],
    parameters: jnp.ndarray,
    wavelength: float,
    refinement_dict: Dict[str, Any],
    symm_strings: List[str],
    information: Dict[str, Any]
) -> pd.DataFrame:
    """Write a fcf file from the results of the refinement for both archival
    purposes and the visualisation of difference electron densities

    Parameters
    ----------
    fcf_path : str
        Path, where the new fcf-file should be written
    fcf_dataset : str
        Dataset name within the fcf file.
    fcf_mode : int
        Can be either 4 or 6 at the moment. See SHELXL documentation
    cell : np.ndarray
        array with the lattice constants (Angstroem, Degree)
    hkl : pd.DataFrame
        pandas DataFrame containing the reflection data. Needs to have at least
        five columns: h, k, l, intensity, esd_int, Additional columns will be
        ignored
    construction_instructions : List[AtomInstructions]
        List of instructions for reconstructing the atomic parameters from the
        list of refined parameters
    parameters : jnp.ndarray
        final refined parameters
    wavelength : float
        Measurement wavelength in Angstroem
    refinement_dict : Dict[str, Any]
        Dictionary with refinement options. For detailed options see refinement 
        function in core
    symm_strings : List[str]
        strings containing the symmetry information of the unit cell in the
        usual cif format
    information : Dict[str, Any]
        Dictionary with additional information, obtained from the refinement.
        the atomic form factors will be read from this dict.

    Returns
    -------
    hkl: pd.DataFrame
        DataFrame with additional columns depending on the fcf mode used

    Raises
    ------
    NotImplementedError
        Selected a fcf mode, which was not implemented
    """
    hkl = hkl.copy()
    cell_mat_m = cell_constants_to_M(*cell)
    constructed_xyz, constructed_uij, constructed_cijk, constructed_dijkl, constructed_occupancies = construct_values(parameters, construction_instructions, cell_mat_m)
    symm_list = [symm_to_matrix_vector(instruction) for instruction in symm_strings]
    symm_mats_r, symm_vecs_t = zip(*symm_list)
    symm_mats_vecs = (np.array(symm_mats_r), np.array(symm_vecs_t))
    cell_mat_f = np.linalg.inv(cell_mat_m).T
    index_vec_h = hkl[['h', 'k', 'l']].values

    intensity = hkl['intensity'].values.copy()
    esd_int = hkl['esd_int'].values.copy()

    structure_factors = np.array(calc_f(
        xyz=constructed_xyz,
        uij=constructed_uij,
        cijk=constructed_cijk,
        dijkl=constructed_dijkl,
        occupancies=constructed_occupancies,
        index_vec_h=index_vec_h,
        cell_mat_f=cell_mat_f,
        symm_mats_vecs=symm_mats_vecs,
        f0j=information['f0j_anom']
    ))

    extinction = get_value_or_default('extinction', refinement_dict)

    if extinction == 'none':
        hkl['intensity'] = np.array(intensity / parameters[0])
        hkl['esd_int'] = np.array(esd_int / parameters[0])
    elif extinction == 'secondary':
        extinction_parameter = get_parameter_index('extinction', refinement_dict)
        i_calc0 = np.abs(structure_factors)**2
        hkl['intensity'] = np.array(intensity / parameters[0] * (1 + parameters[extinction_parameter] * i_calc0))
        hkl['esd_int'] = np.array(esd_int / parameters[0] * (1 + parameters[extinction_parameter] * i_calc0))
    elif extinction == 'shelxl':
        extinction_parameter = get_parameter_index('extinction', refinement_dict)
        i_calc0 = np.abs(structure_factors)**2
        sintheta = np.linalg.norm(np.einsum('xy, zy -> zx', cell_mat_f, index_vec_h), axis=1) / 2 * wavelength
        sintwotheta = 2 * sintheta * np.sqrt(1 - sintheta**2)
        extinction_factors = 0.001 * wavelength**3 / sintwotheta
        hkl['intensity'] = np.array(intensity / parameters[0] * np.sqrt(1 + parameters[extinction_parameter] * extinction_factors * i_calc0))
        hkl['esd_int'] = np.array(esd_int / parameters[0] * np.sqrt(1 + parameters[extinction_parameter] * extinction_factors * i_calc0))
    else:
        raise NotImplementedError('Extinction correction method is not implemented in fcf routine')

    tds = get_value_or_default('tds', refinement_dict)
    if tds == 'Zavodnik':
        tds_indexes = get_parameter_index('tds', refinement_dict)
        
        a_tds = parameters[tds_indexes[0]]
        b_tds = parameters[tds_indexes[1]]
        sin_th_ov_lam = np.linalg.norm(np.einsum('xy, zy -> zx', cell_mat_f, index_vec_h), axis=1) / 2
        hkl['intensity'] /= 1 + a_tds * sin_th_ov_lam**2 + b_tds * sin_th_ov_lam**3
        hkl['esd_int'] /= 1 + a_tds * sin_th_ov_lam**2 + b_tds * sin_th_ov_lam**3
    elif tds == 'none':
        pass
    else: 
        raise NotImplementedError('tds correction is not implemented in fcf routine')

    if fcf_mode == 6:
        dispersion_real = jnp.array([atom.dispersion_real for atom in construction_instructions])
        dispersion_imag = jnp.array([atom.dispersion_imag for atom in construction_instructions])
        f_dash = dispersion_real + 1j * dispersion_imag        

        hkl['abs(f_calc)'] = np.abs(structure_factors)

        # the phases are calculated without dispersion
        structure_factors = np.array(calc_f(
            xyz=constructed_xyz,
            uij=constructed_uij,
            cijk=constructed_cijk,
            dijkl=constructed_dijkl,
            occupancies=constructed_occupancies,
            index_vec_h=index_vec_h,
            cell_mat_f=cell_mat_f,
            symm_mats_vecs=symm_mats_vecs,
            f0j=information['f0j_anom'] - f_dash[None, :, None]
        ))

        hkl['phase_angle'] = np.round(np.rad2deg(np.angle(structure_factors)),3) % 360
        #template = '{h:>4d}{k:>4d}{l:>4d} {intensity:13.2f} {esd_int:13.2f} {abs(f_calc):13.2f} {phase_angle:7.3f}\n'
        template = '{h} {k} {l} {intensity:.3f} {esd_int:.3f} {abs(f_calc):.4f} {phase_angle:.3f}\n'
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
        template = '{h:>4d}{k:>4d}{l:>4d} {i_calc:13.2f} {intensity:13.2f} {esd_int:13.2f} {observed}\n'
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
        f'data_{fcf_dataset}\n',
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
    with open(fcf_path, 'w') as fo:
        fo.write('\n'.join(output))

    return hkl


def entries2atom_string(
    label: str,
    sfac_index: int,
    xyz: np.ndarray,
    uij: np.ndarray,
    occupancy: float
) -> str:
    """Helper function to create a shelxl .res atom string from the given
    parameters

    Parameters
    ----------
    label : str
        Atom label
    sfac_index : int
        SFAC index within shelxl
    xyz : np.ndarray
        fractional coordinates
    uij : np.ndarray
        Anisotropic displacement parameters
    occupancy : float
        occupancies

    Returns
    -------
    str
        The correct atomic string for the SHELXL output
    """
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


def write_res(
    out_res_path: str,
    in_res_path: str,
    cell: jnp.ndarray,
    cell_esd: jnp.ndarray,
    construction_instructions: List[AtomInstructions],
    parameters: jnp.ndarray,
    wavelength: float
):
    """Write a SHELXL .res file for visualisation. Will always convert isotropic
    displacement parameters to the equivalent anistropic ones. Input res is
    still needed to get the LATT, SYMM and SFAC instructions.

    Parameters
    ----------
    out_res_path : str
        Path, where the new .res will be written
    in_res_path : str
        Input res file to copy some entries. Should also work with a .lst file
    cell : jnp.ndarray
        array with the lattice constants (Angstroem, Degree)
    cell_esd : jnp.ndarray
        array with the estimated standard deviation of the lattice constants
        (Angstroem, Degree)
    construction_instructions : List[AtomInstructions]
        List of instructions for reconstructing the atomic parameters from the
        list of refined parameters
    parameters : jnp.ndarray
        final refined parameters
    wavelength : float
        Measurement wavelength in Angstroem
    """
    with open(in_res_path) as fo:
        res_lines = fo.readlines()
    cell_mat_m = cell_constants_to_M(*cell)
    xyzs, uijs, _, _, occs = construct_values(parameters, construction_instructions, cell_mat_m)

    latt_line = [line.strip() for line in res_lines if line.upper().strip().startswith('LATT ')][0]
    symm_lines = [line.strip() for line in res_lines if line.upper().strip().startswith('SYMM ')]
    symm_string = '\n'.join(symm_lines)
    sfac_line = [line.strip() for line in res_lines if line.upper().strip().startswith('SFAC ')][0]
    sfac_elements = [element.capitalize() for element in sfac_line.split()[1:]]
    unit_entries = ' '.join(['99'] * len(sfac_elements))
    sfacs = [sfac_elements.index(instr.element.capitalize()) + 1 for instr in construction_instructions]
    entry_zip = zip(construction_instructions, sfacs, xyzs, uijs, occs)
    atom_lines = '\n'.join([entries2atom_string(inst.name, sfac, xyz, uij, occ) for inst, sfac, xyz, uij, occ in entry_zip])

    output_res = f"""TITL har_out
CELL  {wavelength} {cell[0]:6.4f} {cell[1]:6.4f} {cell[2]:6.4f} {cell[3]:6.4f} {cell[4]:6.4f} {cell[5]:6.4f}
ZERR  999 {cell_esd[0]:6.4f} {cell_esd[1]:6.4f} {cell_esd[2]:6.4f} {cell_esd[3]:6.4f} {cell_esd[4]:6.4f} {cell_esd[5]:6.4f}
{latt_line}
{symm_string}
{sfac_line}
UNIT {unit_entries}
LIST 6
L.S. 0
FMAP 2
WGHT    0.000000
FVAR       {np.sqrt(parameters[0]):8.6f}
{atom_lines}
HKLF 4
END
"""
    with open(out_res_path, 'w') as fo:
        fo.write(output_res)


def value_with_esd(
    values: np.ndarray,
    esds: np.ndarray
) -> Union[List[str], str]:
    """Create string or strings with the values und estimated standard 
    deviation in the format xxx.xx(y). Will round to 6 digits if esd is
    nan or zero. Can only handle one-dimensional arrays. So loop outside for use

    Parameters
    ----------
    values : np.ndarray
        At maximum a one dimensional array containing the values
    esds : np.ndarray
        At maximum a one dimensional array containing the esds

    Returns
    -------
    formatted_strings: Union[List[str], str]
        string or strings that are formatted according to the style
    """
    try:
        assert len(values.shape) == 1, 'Multidimensional array currently not supported'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            indexes = np.isfinite(1 / esds)
            orders = np.floor(np.log10(esds))
            smaller2 = np.full_like(values, False)
            smaller2[indexes] = np.array(esds[indexes]) * 10**(-orders[indexes]) < 2

            orders[np.logical_and(smaller2, orders)] -= 1
            orders[orders > 0] = 0 
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
        # we have only a single value
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
    except ValueError:
        print(values, esds)
        raise


def cif_entry_string(
    name: str,
    value: Union[None, str, float, int],
    string_sign: bool = True
) -> str:
    """Create a cif line from the given values

    Parameters
    ----------
    name : str
        Name of the cif entry without preceding underscore
    value : Union[None, str, float, int]
        Value corresponding to the cif entry in one of the given types. 
    string_sign : bool, optional
        Embed an entry of type cif 'like this', Lots of options in cif file are
        formatted without the string sign if there is a limited number of 
        options. Can also be used to write out numerical values with esd,
        by default True

    Returns
    -------
    str
        Output line for the cif entry

    Raises
    ------
    NotImplementedError
        Unknown variable type for value. Either implement or cast into one of
        the known ones (str with string_sign False should always work)
    """
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


def add_from_cif(
    name: str,
    cif: OrderedDict,
    esd: bool = False,
    string_sign: bool = True
) -> str:
    """Try to add a value from another cif file, read in by ciflike_to_dict

    Parameters
    ----------
    name : str
        Name of the cif entry
    cif : OrderedDict
        The dictionary generated from the read in cif file
    esd : bool, optional
        Value comes with an estimated standard deviation so this is considered
        as well, by default False
    string_sign : bool, optional
        Value is a string and should also be written 'like this', by default
        True

    Returns
    -------
    str
        Formatted entry if the value is present in the cif file. Otherwise the
        entry will be marked with the . for a missing entry.
    """
    #assert not (std and string_sign), 'Cannot be both a string and a val with std'
    if esd:
        std_name = name + '_esd'
        if std_name in cif:
            return cif_entry_string(
                name,
                value_with_esd(cif[name], cif[std_name]),
                False
            )
    try: 
        return cif_entry_string(name, cif[name], string_sign)
    except KeyError:
        return cif_entry_string(name, None)


def cif2atom_type_table_string(
    cif: OrderedDict,
    version: str,
    ishar: bool = True
) -> str:
    """Helper function to create the atom_type table from a given cif

    Parameters
    ----------
    cif : OrderedDict
        The dictionary generated from the read in cif file
    versionmajor : str
        current XHARPy version
    ishar : bool, optional
        refinement is a Hirshfeld Atom Refinement in XHARPy, by default True

    Returns
    -------
    str
        Formatted atom_type table
    """
    table = next(loop for loop in cif['loops'] if 'atom_type_symbol' in loop.columns)
    if ishar:
        table['atom_type_scat_source'] = f'HAR in XHARPy {XHARPY_VERSION}'
    else:
        table['atom_type_scat_source'] = 'International Tables Vol C Tables 4.2.6.8 and 6.1.1.4'
    columns = [column for column in table.columns if not column.endswith('_esd')]
    string = '\nloop_\n _' + '\n _'.join(columns) + '\n'
    template = (" '{atom_type_symbol}' '{atom_type_description}' "
     + '{atom_type_scat_dispersion_real:6.4f} {atom_type_scat_dispersion_imag:6.4f} '
     + "'{atom_type_scat_source}'\n")
    for index, row in table.iterrows():
        string += template.format(**row)
    return string


def cif2space_group_table_string(cif: OrderedDict) -> str:
    """Helper function the generate the space_group table from a given cif
    file

    Parameters
    ----------
    cif : OrderedDict
        The dictionary generated from the read in cif file

    Returns
    -------
    string
        Formatted space_group table
    """
    table = next(loop for loop in cif['loops'] if 'space_group_symop_operation_xyz' in loop.columns)
    columns = [column for column in table.columns if not column.endswith('_esd')]
    string = '\nloop_\n _' + '\n _'.join(columns) + '\n'
    template = " '{space_group_symop_operation_xyz}'\n"
    for _, row in table.iterrows():
        string += template.format(**row)
    return string


def create_atom_site_table_string(
    parameters: jnp.ndarray,
    construction_instructions: List[AtomInstructions],
    cell: jnp.ndarray,
    cell_esd: jnp.ndarray, 
    var_cov_mat: jnp.ndarray, 
    crystal_system: str
) -> str:
    """Helper function to create the atom_site table from the refined parameters

    Parameters
    ----------
    parameters : jnp.ndarray
        final refined parameters
    construction_instructions : List[AtomInstructions]
        list of AtomInstructions used in the refinement.
    cell : jnp.ndarray
        array with the lattice constants (Angstroem, Degree)
    cell_esd : jnp.ndarray
        array with the estimated standard deviation of the lattice constants
        (Angstroem, Degree)
    var_cov_mat : jnp.ndarray
        variance covariance matrix of the final refinement step
    crystal_system : str
        Crystal system of the evaluated structure. Possible values are: 
        'triclinic', 'monoclinic' 'orthorhombic', 'tetragonal', 'hexagonal',
        'trigonal' and 'cubic'. Is considered for the esd calculation.

    Returns
    -------
    str
        Formatted atom_site table

    Raises
    ------
    NotImplementedError
        Non-implemented adp_type
    """
    columns = ['label', 'type_symbol', 'fract_x', 'fract_y', 'fract_z',
               'U_iso_or_equiv', 'adp_type', 'occupancy', 'site_symmetry_order']
    columns = ['atom_site_' + name for name in columns]
    string = '\nloop_\n _' + '\n _'.join(columns) + '\n'
    cell_mat_m = cell_constants_to_M(*cell)
    constructed_xyz, _, _, _, _ = construct_values(parameters, construction_instructions, cell_mat_m)
    constr_xyz_esd, _, _, _, _ = construct_esds(var_cov_mat, construction_instructions)

    for index, (xyz, xyz_esd, instr) in enumerate(zip(constructed_xyz, constr_xyz_esd, construction_instructions)):
        adp_type = instr.uij.adp_type
        occupancy = value_with_esd(
            instr.occupancy.occupancy(parameters),
            instr.occupancy.occupancy_esd(var_cov_mat)
        )
        symmetry_order = instr.occupancy.symmetry_order()
        position_string = ' '.join(value_with_esd(xyz, xyz_esd))
        uiso, uiso_esd = u_iso_with_esd(instr.name, construction_instructions, parameters, var_cov_mat, cell, cell_esd, crystal_system)
        uiso_string = value_with_esd(float(uiso), float(uiso_esd))
        string += f'{instr.name} {instr.element} {position_string} {uiso_string} {adp_type} {occupancy} {symmetry_order}\n'
    return string


def create_aniso_table_string(
    parameters: jnp.ndarray,
    construction_instructions: List[AtomInstructions],
    cell: jnp.ndarray,
    var_cov_mat: jnp.ndarray
) -> str:
    """Create a formatted atom_site_aniso table from the given arguments

    Parameters
    ----------
    parameters : jnp.ndarray
        final refined parameters
    construction_instructions : List[AtomInstructions]
        list of AtomInstructions used in the refinement.
    cell : jnp.ndarray
        array with the lattice constants (Angstroem, Degree)
    var_cov_mat : jnp.ndarray
        variance covariance matrix of the final refinement step

    Returns
    -------
    str
        Formatted atom_site_aniso string
    """
    columns = ['label', 'U_11', 'U_22', 'U_33', 'U_23', 'U_13', 'U_12']
    columns = ['atom_site_aniso_' + name for name in columns]
    string = '\nloop_\n _' + '\n _'.join(columns) + '\n'

    cell_mat_m = cell_constants_to_M(*cell)
    _, uijs, *_ = construct_values(parameters, construction_instructions, cell_mat_m)
    _, uij_esds, *_ = construct_esds(var_cov_mat, construction_instructions)

    for instr, uij, uij_esd in zip(construction_instructions, uijs, uij_esds):
        if instr.uij.adp_type == 'Uani':
            # we have an anisotropic adp
            uij_string = ' '.join(value_with_esd(uij, uij_esd))
            string += f'{instr.name} {uij_string}\n'
    return string


def create_gc3_table_string(
    parameters: jnp.ndarray,
    construction_instructions: List[AtomInstructions],
    cell: jnp.ndarray,
    var_cov_mat: jnp.ndarray
) -> str:
    """Create a formatted atom_site_anharm_GC_C table if any values are
    different from zero

    Parameters
    ----------
    parameters : jnp.ndarray
        final refined parameters
    construction_instructions : List[AtomInstructions]
        list of AtomInstructions used in the refinement.
    cell : jnp.ndarray
        array with the lattice constants (Angstroem, Degree)
    var_cov_mat : jnp.ndarray
        variance covariance matrix of the final refinement step

    Returns
    -------
    str
        Formatted atom_site_anharm_GC_C table if any value different from zero,
        otherwise an empty string is returned
    """
    columns = ['label', '111', '222', '333', '112', '122', '113', '133', '223', '233', '123']
    columns = ['atom_site_anharm_GC_C_' + name for name in columns]

    string = '\nloop_\n _' + '\n _'.join(columns) + '\n'

    cell_mat_m = cell_constants_to_M(*cell)
    _, _, cijks, *_ = construct_values(parameters, construction_instructions, cell_mat_m)
    _, _, cijk_esds, *_ = construct_esds(var_cov_mat, construction_instructions)

    table_needed = False
    for instr, cijk, cijk_esd in zip(construction_instructions, cijks, cijk_esds):
        if np.sum(np.abs(cijk)) > 1e-30:
            table_needed = True
            cijk_string = ' '.join(value_with_esd(cijk, cijk_esd))
            cijk_string = '\n'.join(textwrap.wrap(cijk_string, width=75))
            string += f'{instr.name} {cijk_string}\n'
    if table_needed:
        return string
    else:
        return ''


def create_gc4_table_string(
    parameters: jnp.ndarray,
    construction_instructions: List[AtomInstructions],
    cell: jnp.ndarray,
    var_cov_mat: jnp.ndarray
) -> str:
    """Create a formatted atom_site_anharm_GC_D table if any values are
    different from zero

    Parameters
    ----------
    parameters : jnp.ndarray
        final refined parameters
    construction_instructions : List[AtomInstructions]
        list of AtomInstructions used in the refinement.
    cell : jnp.ndarray
        array with the lattice constants (Angstroem, Degree)
    var_cov_mat : jnp.ndarray
        variance covariance matrix of the final refinement step

    Returns
    -------
    str
        Formatted atom_site_anharm_GC_D table if any value different from zero,
        otherwise an empty string is returned
    """
    columns = ['label', '1111', '2222', '3333', '1112', '1222', '1113', '1333', '2223', '2333',
               '1122', '1133', '2233', '1123', '1223', '1233']
    columns = ['atom_site_anharm_GC_D_' + name for name in columns]

    string = '\nloop_\n _' + '\n _'.join(columns) + '\n'

    cell_mat_m = cell_constants_to_M(*cell)
    _, _, _, dijkls, *_ = construct_values(parameters, construction_instructions, cell_mat_m)
    _, _, _, dijkl_esds, *_ = construct_esds(var_cov_mat, construction_instructions)

    table_needed = False
    for instr, dijkl, dijk_esd in zip(construction_instructions, dijkls, dijkl_esds):
        if np.sum(np.abs(dijkl)) > 1e-30:
            table_needed = True
            dijkl_string = ' '.join(value_with_esd(dijkl, dijk_esd))
            dijkl_string = '\n'.join(textwrap.wrap(dijkl_string, width=75))
            string += f'{instr.name} {dijkl_string}\n'
    if table_needed:
        return string
    else:
        return ''

def site_symm2mat_vec(
    code: str,
    symm_mats_vecs: Tuple[np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Covert a cif site_symmetry string into the corresponding symmetry matrix
    and translation vector

    Parameters
    ----------
    code : str
        site_symm code from a cif file, e.g. 2_565
    symm_mats_vecs : Tuple[np.ndarray, np.ndarray]
        size (K, 3, 3) array of symmetry matrices and (K, 3) array of translation
        vectors for all symmetry elements in the unit cell

    Returns
    -------
    symm_mat : np.ndarray
        size (3, 3) symmetry matrix
    trans_vec :  np.ndarray
        size (3) array contraining the translation from symmetry and additional
        translation from code
    """
    if code == '.':
        return np.eye(3), np.zeros(3)
    code_split = code.split('_')
    symm_mats, symm_vecs = symm_mats_vecs
    if len(code_split) == 1:
        symm_index = int(code) - 1
        return symm_mats[symm_index], np.zeros(3)
    else:
        symm_index_str, trans_code = code_split
        t_add = np.array([
            float(trans_code[0]) - 5,
            float(trans_code[1]) - 5,
            float(trans_code[2]) - 5,
        ])
        symm_index = int(symm_index_str) - 1
        return symm_mats[symm_index], symm_vecs[symm_index] + t_add


def create_distance_table(
    bonds: List[Tuple[str, str, str]],
    construction_instructions: List[AtomInstructions],
    parameters: jnp.ndarray,
    var_cov_mat: jnp.ndarray,
    cell: jnp.ndarray,
    cell_esd: jnp.ndarray,
    crystal_system: str,
    symm_mats_vecs: Tuple[np.ndarray, np.ndarray]
) -> str:
    """Create a distance table for the given parameters

    Parameters
    ----------
    bonds : List[Tuple[str, str, str]]
        List with tuples of atom names and a corresponding cif symmetry code for
        the second atom. Distances between the atoms in the tuple are output to
        the bond table.
    construction_instructions : List[AtomInstructions]
        list of AtomInstructions used in the refinement.
    parameters : jnp.ndarray
        final refined parameters
    var_cov_mat : jnp.ndarray
        variance covariance matrix of the final refinement step
    cell : jnp.ndarray
        array with the lattice constants (Angstroem, Degree)
    cell_esd : jnp.ndarray
        array with the estimated standard deviation of the lattice constants
        (Angstroem, Degree)
    crystal_system : str
        Crystal system of the evaluated structure. Possible values are: 
        'triclinic', 'monoclinic' 'orthorhombic', 'tetragonal', 'hexagonal',
        'trigonal' and 'cubic'. Is considered for the esd calculation.

    Returns
    -------
    str
        Formatted geom_bond table
    """
    if len(bonds) == 0:
        return ''
    columns =  ['geom_bond_atom_site_label_1',
                'geom_bond_atom_site_label_2',
                'geom_bond_distance',
                'geom_bond_site_symmetry_2']
    string = '\nloop_\n _' + '\n _'.join(columns) + '\n'
    dist_symm_mats_vecs = [
        site_symm2mat_vec(bond[2], symm_mats_vecs) for bond in bonds
    ]
    distances_esds = [
        distance_with_esd(
            bond[0],
            bond[1],
            construction_instructions,
            parameters, 
            var_cov_mat, 
            cell, 
            cell_esd, 
            crystal_system,
            symm2[0],
            symm2[1]
        ) for bond, symm2 in zip(bonds, dist_symm_mats_vecs)
    ]
    distances, distance_esds = zip(*distances_esds)
    distance_esds = np.array(distance_esds)
    distance_esds[distance_esds < 1e-12] = np.nan 
    distance_strings = value_with_esd(np.array(distances), distance_esds)
    string += ''.join([
        f'{atom1} {atom2} {distance_string} {symm_code}\n' 
        for (atom1, atom2, symm_code), distance_string in zip(bonds, distance_strings)
    ])
    return string


def create_angle_table(
    angles: List[Tuple[str, str, str, str, str]],
    construction_instructions: List[AtomInstructions],
    parameters: jnp.ndarray,
    var_cov_mat: jnp.ndarray,
    cell: jnp.ndarray,
    cell_esd: jnp.ndarray,
    crystal_system: str,
    symm_mats_vecs: Tuple[np.ndarray, np.ndarray]
) -> str:
    """Create a angle table for the given parameters

    Parameters
    ----------
    bonds : List[Tuple[str, str, str, str, str]]
        List with tuples of atom names. Angles spanned the atoms 
        are output to the angle table and two cif symmetry codes for the
        first and third atom of the angle
    construction_instructions : List[AtomInstructions]
        list of AtomInstructions used in the refinement.
    parameters : jnp.ndarray
        final refined parameters
    var_cov_mat : jnp.ndarray
        variance covariance matrix of the final refinement step
    cell : jnp.ndarray
        array with the lattice constants (Angstroem, Degree)
    cell_esd : jnp.ndarray
        array with the estimated standard deviation of the lattice constants
        (Angstroem, Degree)
    crystal_system : str
        Crystal system of the evaluated structure. Possible values are: 
        'triclinic', 'monoclinic' 'orthorhombic', 'tetragonal', 'hexagonal',
        'trigonal' and 'cubic'. Is considered for the esd calculation.

    Returns
    -------
    str
        Formatted geom_angle table
    """
    columns =  ['geom_angle_atom_site_label_1',
                'geom_angle_atom_site_label_2',
                'geom_angle_atom_site_label_3',
                'geom_angle',
                'geom_angle_site_symmetry_1',
                'geom_angle_site_symmetry_3']
    if len(angles) == 0:
        return ''
    angle_symm_mats_vecs1 = [
        site_symm2mat_vec(angle[3], symm_mats_vecs) for angle in angles
    ]
    angle_symm_mats_vecs3 = [
        site_symm2mat_vec(angle[4], symm_mats_vecs) for angle in angles
    ]
    string = '\nloop_\n _' + '\n _'.join(columns) + '\n'
    angles_esds = [
        angle_with_esd(
            angle[0],
            angle[1],
            angle[2],
            construction_instructions,
            parameters,
            var_cov_mat,
            cell,
            cell_esd,
            crystal_system,
            symm1[0],
            symm1[1],
            symm3[0],
            symm3[1]
        ) for angle, symm1, symm3 in zip(angles, angle_symm_mats_vecs1, angle_symm_mats_vecs3)]
    angle_vals, angle_esds = zip(*angles_esds)
    angle_esds = np.array(angle_esds)
    angle_esds[angle_esds < 1e-12] = np.nan # account for numerical noise 

    angle_strings = value_with_esd(np.array(angle_vals), np.array(angle_esds))
    string += ''.join([
        f'{atom1} {atom2} {atom3} {angle_string} {symm_code1} {symm_code3}\n'
        for (atom1, atom2, atom3, symm_code1, symm_code3), angle_string in zip(angles, angle_strings)
    ])
    return string


def create_fcf4_table(
    index_vec_h: jnp.ndarray,
    structure_factors: jnp.ndarray,
    intensity: jnp.ndarray,
    esd_int: jnp.ndarray,
    scaling: float
)-> str:
    """Create a formatted fcf4 table for output in cif file

    Parameters
    ----------
    index_vec_h : jnp.ndarray
        size (H, 3) array of Miller indicees of observed reflections
    structure_factors : jnp.ndarray
        size (H)-sized array with complex structure factors for each reflection
    intensity : jnp.ndarray
        size (H) array of observed reflection intensities
    esd_int : jnp.ndarray
        Estimated standard deviation of the observed reflection intensites
    scaling : float
        overall scaling factor

    Returns
    -------
    str
        formatted fcf4 table
    """
    columns =  ['refln_index_h',
                'refln_index_k',
                'refln_index_l',
                'refln_F_squared_calc',
                'refln_F_squared_meas',
                'refln_F_squared_sigma']
    string = '\nloop_\n _' + '\n _'.join(columns) + '\n'

    for (h, k, l), i_calc, i_meas, esd_meas in zip(index_vec_h, np.abs(structure_factors)**2, intensity / scaling, esd_int / scaling):
        string += f'{h:>4d}{k:>4d}{l:>4d}{i_calc:14.2f} {i_meas:14.2f}{esd_meas:14.2f}\n'
    return string

def create_diff_density_entries(symm_mats_vecs, index_vec_h, scaled_intensity, structure_factors, cell_mat_m, dims=np.array([53, 53, 53])):
    """This function currently only adds empty entries that need to be filled in
    with cctbx as the current implementation produces significantly different
    results
    """

    """
    #TODO Fix this
    symm_mats, symm_vecs = symm_mats_vecs
    scaled_intensity = np.array(scaled_intensity)
    scaled_intensity[scaled_intensity < 0] = 0
    deltafs2 = np.sqrt(scaled_intensity) - np.abs(structure_factors)
    xxx, yyy, zzz = np.meshgrid(*[np.linspace(0, 1, dim, endpoint=False) for dim in dims], indexing='ij')
    xyz = np.array([xxx, yyy, zzz])
    angles = np.angle(structure_factors)
    diff = np.zeros_like(xxx)
    for hkl_ind, deltaf, angle in zip(index_vec_h, deltafs2, angles):
        hkl_symm = np.einsum('x, axy -> ay', hkl_ind, symm_mats)
        hkl_symm, unique_indexes = np.unique(hkl_symm, axis=0, return_index=True)
        shifts = 2 * np.pi * np.einsum('x, ax -> a', hkl_ind, symm_vecs)[unique_indexes]
        full_angle = 2 * np.pi * np.einsum('ax, xijk -> aijk', hkl_symm, xyz) + angle - shifts[:, None, None, None]
        
        diff += deltaf * np.sum(np.cos(full_angle), axis=0)
    diff = diff / np.linalg.det(cell_mat_m)

    return '\n'.join([
        cif_entry_string('refine_diff_density_max', float(np.round(np.max(diff), 4))),
        cif_entry_string('refine_diff_density_min', float(np.round(np.min(diff), 4))),
        cif_entry_string('refine_diff_density_rms', float(np.round(np.std(diff), 4)))
    ])
    """
    return '\n'.join([
        cif_entry_string('refine_diff_density_max', '.', False),
        cif_entry_string('refine_diff_density_min', '.', False),
        cif_entry_string('refine_diff_density_rms', '.', False)
    ])


def create_extinction_entries(
    parameters: jnp.ndarray,
    var_cov_mat: jnp.ndarray,
    refinement_dict: Dict[str, Any]
) -> str:
    """Create the set of extinction related entries as string output

    Parameters
    ----------
    parameters : jnp.ndarray
        final refined parameters
    var_cov_mat : jnp.ndarray
        variance covariance matrix of the final refinement step
    refinement_dict : Dict[str, Any]
        Dictionary with refinement options. For detailed options see refinement 
        function in core
    Returns
    -------
    str
        Formatted string for output

    Raises
    ------
    NotImplementedError
        Used extinction method is not implemented in this output function
    """
    extinction = get_value_or_default('extinction', refinement_dict)
    if extinction == 'none':
        method = 'none'
        coeff = '.'
    else:
        extinction_parameter = get_parameter_index('extinction', refinement_dict)
        exti = parameters[extinction_parameter]
        esd = np.sqrt(var_cov_mat[extinction_parameter, extinction_parameter])
        coeff = value_with_esd(np.array([exti]), np.array([esd]))[0]
        if extinction == 'shelxl':
            method = 'SHELXL-2018/3 (Sheldrick 2018)'
        elif extinction == 'secondary':
            method = 'Zachariasen'
        else:
            raise NotImplementedError('This extinction is not implemeted in io')
    entries = [
        cif_entry_string('refine_ls_extinction_coef', coeff, False),
        cif_entry_string('refine_ls_extinction_method', method , False)
    ]
    if extinction == 'shelxl':
        entries.append(cif_entry_string('refine_ls_extinction_expression', 'Fc^*^=kFc[1+0.001xFc^2^\l^3^/sin(2\q)]^-1/4^'
))
    return '\n'.join(entries)


def generate_core_refinement_string(refinement_dict, parameters, var_cov_mat):
    core = get_value_or_default('core', refinement_dict)
    if core == 'combine':
        return """  - Core density was not treated separately from the valence density"""
    elif core == 'constant':
        return """  - Frozen core density was integrated separately on a spherical
    grid and added to the partitioned valence density
  - Core density was always fully assigned to the respective atom"""
    elif core == 'scale':
        core_parameter = get_parameter_index('core', refinement_dict)
        core_str = value_with_esd(float(parameters[core_parameter]),
                                  float(np.sqrt(var_cov_mat[core_parameter, core_parameter])))
        return f"""  - Frozen core density was integrated separately on a spherical grid
  - An overall core scaling factor was refined to {core_str}"""
    else:
        raise NotImplementedError('Core treatment is not implemented in generate_core_refinement_string')

def write_cif(
    output_cif_path: str,
    cif_dataset: Union[str, int],
    shelx_cif_path: str,
    shelx_dataset: Union[str, int],
    cell: jnp.ndarray,
    cell_esd: jnp.ndarray,
    symm_mats_vecs: Tuple[jnp.ndarray, jnp.ndarray],
    hkl: pd.DataFrame,
    construction_instructions: List[AtomInstructions],
    parameters: jnp.ndarray,
    var_cov_mat: jnp.ndarray,
    refinement_dict: Dict[str, Any],
    computation_dict: Dict[str, Any],
    information: Dict[str, Any],
    source_cif_path: str = None,
    source_dataset: Union[str, int] = None,
):
    """Write a cif file from the given parameters. Currently only works for gpaw
    and iam refinements.

    Parameters
    ----------
    output_cif_path : str
        path where the new cif file is output
    cif_dataset : Union[str, int]
        name of dataset in cif file
    shelx_cif_path : str
        Path to the shelx cif used for the refinement. For the time being is
        used to copy some values, where the calculation is not implemented yet
    shelx_dataset : Union[str, int]
        dataset within the shelx cif file
    cell : jnp.ndarray
        array with the lattice constants (Angstroem, Degree)
    cell_esd : jnp.ndarray
        array with the estimated standard deviation of the lattice constants
        (Angstroem, Degree)
    symm_mats_vecs : Tuple[jnp.ndarray, jnp.ndarray]
        size (K, 3, 3) array of symmetry matrices and (K, 3) array of translation
        vectors for all symmetry elements in the unit cell
    hkl : pd.DataFrame
        pandas DataFrame containing the reflection data. Needs to have at least
        five columns: h, k, l, intensity, esd_int, Additional columns will be
        ignored
    construction_instructions : List[AtomInstructions]
        List of instructions for reconstructing the atomic parameters from the
        list of refined parameters
    parameters : jnp.ndarray
        final refined parameters
    var_cov_mat : jnp.ndarray
        [description]
    refinement_dict : Dict[str, Any]
        Dictionary with refinement options. For detailed options see refinement 
        function in core
    computation_dict : Dict[str, Any]
        Dict with options of the to the f0j_source.
    information : Dict[str, Any]
        Dictionary with additional information, obtained from the refinement.
        the atomic form factors will be read from this dict.
    source_cif_path : str, optional
        Additional cif that will be searched for crystal and measurement
        information. If not given the shelx cif will be tried instead
    source_dataset : Union[str, int], optional
        Dataset to use in the source_cif
    """

    if source_cif_path is None:
        source_cif_path = shelx_cif_path
    if source_dataset is None:
        source_dataset = shelx_dataset

    shelx_cif = ciflike_to_dict(shelx_cif_path, shelx_dataset)
    source_cif = ciflike_to_dict(source_cif_path, source_dataset)

    crystal_system = shelx_cif['space_group_crystal_system']
    hkl = hkl.copy()
    hkl['strong_condition'] = hkl['intensity'] / hkl['esd_int'] > 2
    index_vec_h = hkl[['h', 'k', 'l']].values
    intensity = hkl['intensity'].values
    esd_int = hkl['esd_int'].values
    cell_mat_m = cell_constants_to_M(*cell)
    cell_mat_f = np.linalg.inv(cell_mat_m).T
    f0j_source = get_value_or_default('f0j_source', refinement_dict)

    ishar = f0j_source != 'iam'
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
        f0j=information['f0j_anom']
    ))
    if f0j_source == 'iam':
        from .f0j_sources.iam_source import generate_cif_output
    elif f0j_source == 'gpaw':
        from .f0j_sources.gpaw_source import generate_cif_output
    elif f0j_source == 'gpaw_mpi':
        from .f0j_sources.gpaw_mpi_source import generate_cif_output
    elif f0j_source == 'gpaw_spherical':
        from .f0j_sources.gpaw_spherical_source import generate_cif_output
    elif f0j_source == 'qe':
        from .f0j_sources.qe_source import generate_cif_output
    elif f0j_source == 'tsc_file':
        from .f0j_sources.tsc_file_source import generate_cif_output
    elif f0j_source == 'nosphera2_orca':
        from .f0j_sources.nosphera2_orca_source import generate_cif_output
    elif f0j_source == 'custom_function':
        from .f0j_sources.custom_function_source import generate_cif_output
    else:
        raise NotImplementedError('This f0j source has not implemented "generate_cif_output" method')

    refinement_string = """  - Structure optimisation was done using derivatives
    calculated with the python package JAX and
    BFGS minimisation in scipy.optimize.minimize"""

    refinement_string += '\n' + generate_core_refinement_string(
       refinement_dict,
       parameters,
       var_cov_mat
    )
    refinement_string += '\n' + generate_cif_output(computation_dict)

    if source_cif.get('exptl_crystal_description', None) is not None and 'sphere' in source_cif['exptl_crystal_description']:
        crystal_dimension = add_from_cif('exptl_crystal_size_rad', source_cif)
    else:
        crystal_dimension = '\n'.join([
            add_from_cif('exptl_crystal_size_max', source_cif),
            add_from_cif('exptl_crystal_size_mid', source_cif),
            add_from_cif('exptl_crystal_size_min', source_cif)
        ])

    quality_dict = calculate_quality_indicators(
        cell,
        symm_mats_vecs,
        hkl,
        construction_instructions,
        parameters,
        source_cif['diffrn_radiation_wavelength'],
        refinement_dict,
        information,
    )

    bond_table = next(loop for loop in shelx_cif['loops'] if 'geom_bond_distance' in loop.columns)
    bonds = [
        (line['geom_bond_atom_site_label_1'],
        line['geom_bond_atom_site_label_2'],
        line['geom_bond_site_symmetry_2']) for _, line in bond_table.iterrows()
    ]

    angle_table = next(loop for loop in shelx_cif['loops'] if 'geom_angle' in loop.columns)
    angles = [
        (
            line['geom_angle_atom_site_label_1'],
            line['geom_angle_atom_site_label_2'],
            line['geom_angle_atom_site_label_3'],
            line['geom_angle_site_symmetry_1'],
            line['geom_angle_site_symmetry_3']
        ) for _, line in angle_table.iterrows()
    ]
    lines = [
        f'\ndata_{cif_dataset}\n',
        cif_entry_string('audit_creation_method', f'XHARPY {XHARPY_VERSION}'),
        add_from_cif('chemical_name_systematic', source_cif),
        add_from_cif('chemical_name_common', source_cif),
        add_from_cif('chemical_melting_point', source_cif),
        add_from_cif('chemical_formula_moiety', source_cif),
        add_from_cif('chemical_formula_sum', source_cif),
        add_from_cif('chemical_formula_weight', source_cif),
        cif2atom_type_table_string(shelx_cif, XHARPY_VERSION, ishar),
        add_from_cif('space_group_crystal_system', shelx_cif),
        add_from_cif('space_group_IT_number', shelx_cif),
        add_from_cif('space_group_name_H-M_alt', shelx_cif),
        add_from_cif('space_group_name_Hall', shelx_cif),
        cif2space_group_table_string(shelx_cif),
        add_from_cif('cell_length_a', shelx_cif, esd=True),
        add_from_cif('cell_length_b', shelx_cif, esd=True),
        add_from_cif('cell_length_c', shelx_cif, esd=True),
        add_from_cif('cell_angle_alpha', shelx_cif, esd=True),
        add_from_cif('cell_angle_beta', shelx_cif, esd=True),
        add_from_cif('cell_angle_gamma', shelx_cif, esd=True),
        add_from_cif('cell_volume', shelx_cif, esd=True),
        add_from_cif('cell_formula_units_Z', shelx_cif),
        add_from_cif('cell_measurement_temperature', source_cif, esd=True),
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
        add_from_cif('diffrn_ambient_temperature', source_cif, esd=True),
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
        add_from_cif('diffrn_reflns_theta_min', shelx_cif),
        add_from_cif('diffrn_reflns_theta_max', shelx_cif),
        add_from_cif('diffrn_reflns_theta_full', shelx_cif),
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
        cif_entry_string('computing_structure_refinement', f'xHARPY {XHARPY_VERSION}'),
        cif_entry_string('computing_molecular_graphics', None),
        cif_entry_string('computing_publication_material', f'xHARPY {XHARPY_VERSION}'),
        '',
        cif_entry_string('atom_sites_solution_hydrogens', 'difmap', False),
        '',
        cif_entry_string('refine_special_details', refinement_string),
        cif_entry_string('refine_ls_structure_factor_coef', 'Fsqd', False),
        cif_entry_string('refine_ls_matrix_type', 'full'), # TODO is full?
        cif_entry_string('refine_ls_weighting_scheme', 'sigma', False),
        cif_entry_string('refine_ls_weighting_details', 'w=1/[\s^2^(Fo^2^)]'),
        cif_entry_string('refine_ls_hydrogen_treatment', 'refall', False),
        create_extinction_entries(parameters, var_cov_mat, refinement_dict),
        cif_entry_string('refine_ls_number_reflns', len(hkl)),
        cif_entry_string('refine_ls_number_parameters', len(parameters)),
        cif_entry_string('refine_ls_number_restraints', 0),
        cif_entry_string('refine_ls_R_factor_all', float(np.round(quality_dict['R(F)'], 4))),
        cif_entry_string('refine_ls_R_factor_gt', float(np.round(quality_dict['R(F)(I>2s)'], 4))),
        cif_entry_string('refine_ls_wR_factor_ref', float(np.round(quality_dict['wR(F^2)'], 4))),
        cif_entry_string('refine_ls_wR_factor_gt', float(np.round(quality_dict['wR(F^2)(I>2s)'], 4))),
        cif_entry_string('refine_ls_goodness_of_fit_ref', float(np.round(quality_dict['GOF'], 3))),
        cif_entry_string('refine_ls_shift/su_max', float(np.round(np.max(information['shift_ov_su'][0]), 3))),
        cif_entry_string('refine_ls_shift/su_mean', float(np.round(np.mean(information['shift_ov_su'][0]), 3))),
        create_atom_site_table_string(parameters, construction_instructions, cell, cell_esd, var_cov_mat, crystal_system),
        create_aniso_table_string(parameters, construction_instructions, cell, var_cov_mat),
        create_gc3_table_string(parameters, construction_instructions, cell, var_cov_mat),
        create_gc4_table_string(parameters, construction_instructions, cell, var_cov_mat),
        cif_entry_string('geom_special_details', """All esds are estimated using the full variance-covariance matrix.
Correlations between cell parameters are taken into account in the 
calculation of derivatives used for the error propagation to the esds
of U(iso), distances and angles. Otherwise, the esds of the cell
parameters are assumed to be independent."""),
        create_distance_table(bonds, construction_instructions, parameters, var_cov_mat, cell, cell_esd, crystal_system, symm_mats_vecs),
        create_angle_table(angles, construction_instructions, parameters, var_cov_mat, cell, cell_esd, crystal_system, symm_mats_vecs),
        create_diff_density_entries(symm_mats_vecs, index_vec_h, intensity/parameters[0], structure_factors, cell_mat_m),
        create_fcf4_table(index_vec_h, structure_factors, intensity, esd_int, parameters[0])
    ]
    with open(output_cif_path, 'w') as fo:
        fo.write('\n'.join(lines).replace('\n\n\n', '\n\n'))


def add_density_entries_from_fcf(
    cif_path: str,
    fcf6_path: str,
):
    """Adds the density entries to a cif file starting from the fcf6 file.
    This is necessary because so far the difference density calculation 
    is not working properly. This introduces a dependency on the rather
    large cctbx/iotbx library

    Parameters
    ----------
    cif_path : str
        path to the cif-file to be completed
    fcf6_path : str
        path to the fcf6 file to be used for the difference electron density
        calculation
    """
    from iotbx import reflection_file_reader
    reader = reflection_file_reader.cif_reader(fcf6_path)
    arrays = reader.build_miller_arrays()[next(iter(reader.build_miller_arrays()))]
    fobs = arrays['_refln_F_squared_meas'].f_sq_as_f()
    fcalc = arrays['_refln_F_calc']
    diff = fobs.f_obs_minus_f_calc(1.0, arrays['_refln_F_calc'])
    diff_map = diff.fft_map()
    diff_map.apply_volume_scaling()
    stats = diff_map.statistics()
    diff_max = f'{stats.max():17.4f}'
    diff_min = f'{stats.min():17.4f}'
    diff_sigma = f'{stats.sigma():17.4f}'
    with open(cif_path, 'r') as fo:
        content = fo.read()
    content = re.sub(r'(?<=_refine_diff_density_max)\s+([\d\.\-]+)', diff_max, content)
    content = re.sub(r'(?<=_refine_diff_density_min)\s+([\d\.\-]+)', diff_min, content)
    content = re.sub(r'(?<=_refine_diff_density_rms)\s+([\d\.\-]+)', diff_sigma, content)
    with open(cif_path, 'w') as fo:
        fo.write(content)

def f0j2tsc(
    file_name: str,
    f0j: np.ndarray,
    construction_instructions: List[AtomInstructions],
    symm_mats_vecs: Tuple[np.ndarray, np.ndarray],
    index_vec_h: np.ndarray,
    remove_anom: bool = True
):
    """Write a tsc file from the atomic form factor array
    For the format see: https://arxiv.org/pdf/1911.08847.pdf.

    Parameters
    ----------
    file_name : str
        Path to file, where the .tsc should be written
    f0j : np.ndarray
        size (K, N, H) array of atomic form factors for all reflections and symmetry
        generated atoms within the unit cells. Atoms on special positions are 
        present multiple times and have the atomic form factor of the full atom.
        Can be obtained from a calc_f0j function or the information dict of a 
        refinement
    construction_instructions : List[AtomInstructions]
        List of instructions for reconstructing the atomic parameters from the
        list of refined parameters
    symm_mats_vecs : Tuple[np.ndarray, np.ndarray]
        size (K, 3, 3) array of symmetry  matrices and (K, 3) array of
        translation vectors for all symmetry elements in the unit cell
    index_vec_h : np.ndarray
        size (H) vector containing Miller indicees of the measured reflections
    remove_anom : bool, optional
        Determines, whether the dispersion correction should be subtracted. 
        Should be True if you obtained the f0j values from a refinement,
        Should be False if you obtained them from a calc_f0j function, by
        default True
    """

    if remove_anom:
        dispersion_real = jnp.array([atom.dispersion_real for atom in construction_instructions])
        dispersion_imag = jnp.array([atom.dispersion_imag for atom in construction_instructions])
        f_dash = dispersion_real + 1j * dispersion_imag
        f0j -= f_dash[None, :, None]

    labels = [instr.name for instr in construction_instructions]

    hkl_df = pd.DataFrame(columns=['refl_h', 'refl_k', 'refl_l', *labels])

    df_list = []

    for symm_matrix, _, f0j_slice in zip(*symm_mats_vecs, f0j):
        hrot, krot, lrot = np.einsum('zx, xy -> zy', index_vec_h, symm_matrix).T.astype(np.int64)

        new_df = pd.DataFrame({
            'refl_h': hrot,
            'refl_k': krot,
            'refl_l': lrot
        })

        for label, column in zip(labels, f0j_slice):
            new_df[label] = column
            
        df_list.append(new_df)
    
    hkl_df = pd.concat(df_list, ignore_index=True)
        
    hkl_df = hkl_df.drop_duplicates(subset=['refl_h', 'refl_k', 'refl_l'], ignore_index=True)

    complex_entries = [' '.join([f'{np.real(val):10.8e},{np.imag(val):10.8e}' for val in row])
                        for row in  hkl_df.iloc[:, 3:].values]
    hkl_entries = [f'{row[0]} {row[1]} {row[2]} ' for row in hkl_df[['refl_h', 'refl_k', 'refl_l']].values]
    out_data = '\n'.join([hkl_entry + complex_entry for hkl_entry, complex_entry in zip(hkl_entries, complex_entries)])

    header = [
        'TITLE: Atomic Form Factors from XHARPY',
        'SYMM: expanded',
        'AD: FALSE',
        f'SCATTERERS: {" ".join(labels)}',
        'DATA:'
    ]
    with open(file_name, 'w') as fo:
        fo.write('\n'.join(header))
        fo.write('\n')
        fo.write(out_data)

def cif2tsc(
    tsc_path: str, 
    cif_path: str, 
    cif_dataset: Union[str, int], 
    export_dict: Dict[str, Any], 
    computation_dict: Dict[str, Any],
    tsc_core_path: str = None,
) -> None:
    """Can be used to create a tsc file directly from a given cif file
    and the necessary options. 

    Parameters
    ----------
    tsc_path : str
        Path where the new tsc file is meant to be written
    cif_path : str
        Path to the input cif file
    cif_dataset : Union[str, int]
        indentifier of dataset in cif file. If this parameter is a string, the 
        function will search for data\_*cif_dataset* and use the values.
        An integer will be interpreted as an index, starting with 0 as usual in
        python.
    export_dict : Dict[str, Any]
        Dictionary with options for the .tsc export

        - f0j_source (str) : Can be one of the implemented sources for atomic
          form factors. The most common options are 'gpaw', 'gpaw_mpi' and 'qe'
        - core (str): can be either 'constant', which means the core densitsy 
          will be evaluated on a separate spherical grid and assigned to the 
          source atom completely, or 'combine' in which case the core density is
          expanded and partitioned in the same way the valence density is
        - core_io (Tuple[str, str]):
          Expects a tuple where the first entry can be 'save', 'load', 'none'
          which is the action that is taken with the core density. The 
          second argument in the tuple is the filename, to which the core
          density is saved to or loaded from 
        - resolution_limit (float) : resolution limit in Angstrom up to which
          the atomic form factors are evaluated, by default 0.40

    computation_dict : Dict[str, Any]
        Dict with options that are passed on to the f0j_source. See the 
        individual calc_f0j functions for a more detailed description

    Raises
    ------
    NotImplementedError
        f0j_source not implemented
    ValueError
        Tried to load core density, but core density does not have the same 
        number of entries as the hkl indicees
    NotImplementedError
        Type of core description not implemented
    """
    f0j_source = get_value_or_default('f0j_source', export_dict)
    core = get_value_or_default('core', export_dict)
    core_io, core_file = get_value_or_default('core_io', export_dict)
    reslim = export_dict.get('resolution_limit', 0.40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atom_table, cell, cell_std, symm_mats_vecs, symm_strings, wavelength = cif2data(cif_path, cif_dataset)
    atom_table['type_scat_dispersion_real'] = 0.0
    atom_table['type_scat_dispersion_imag'] = 0.0

    cell_mat_m = cell_constants_to_M(*cell)
    cell_mat_f = np.linalg.inv(cell_mat_m).T

    a_star, b_star, c_star = np.linalg.norm(cell_mat_f, axis=1)
    hmax = int(np.ceil(1 / reslim / a_star)) + 1
    kmax = int(np.ceil(1 / reslim / b_star)) + 1
    lmax = int(np.ceil(1 / reslim / c_star)) + 1
    h, k, l = np.meshgrid(np.arange(-hmax, hmax + 1), np.arange(-kmax, kmax + 1), np.arange(-lmax, lmax + 1))
    index_vec_h = np.array([h.ravel(), k.ravel(), l.ravel()]).T
    index_vec_h = index_vec_h[calc_sin_theta_ov_lambda(cell_mat_f, index_vec_h) <= 0.5 / reslim].copy()

    construction_instructions, parameters = create_construction_instructions(
        atom_table, {}, {}, cell
    )

    if f0j_source == 'gpaw':
        from .f0j_sources.gpaw_source import calc_f0j, calc_f0j_core
    elif f0j_source == 'iam':
        from .f0j_sources.iam_source import calc_f0j, calc_f0j_core
    elif f0j_source == 'gpaw_spherical':
        from .f0j_sources.gpaw_spherical_source import calc_f0j, calc_f0j_core
    elif f0j_source == 'qe':
        from .f0j_sources.qe_source import calc_f0j, calc_f0j_core
    elif f0j_source == 'gpaw_mpi':
        from .f0j_sources.gpaw_mpi_source import calc_f0j, calc_f0j_core
    elif f0j_source == 'tsc_file':
        from .f0j_sources.tsc_file_source import calc_f0j, calc_f0j_core
    elif f0j_source == 'nosphera2_orca':
        from .f0j_sources.nosphera2_orca_source import calc_f0j, calc_f0j_core
    elif f0j_source == 'custom_function':
        from .f0j_sources.custom_function_source import calc_f0j, calc_f0j_core
    else:
        raise NotImplementedError('Unknown type of f0j_source')

    if f0j_source in ('iam') and core != 'combine':
        warnings.warn('core description is not possible with this f0j source')
    if core == 'constant':
        print('Calculating core density')
        if core_io == 'load':
            with open(core_file, 'rb') as fo:
                f0j_core = pickle.load(fo)
            if index_vec_h.shape[0] != f0j_core.shape[1]:
                raise ValueError('The loaded core atomic form factors do not match the number of reflections')
            print('  loaded core atomic form factors from disk')
        elif f0j_source == 'qe':
            f0j_core, computation_dict = calc_f0j_core(
                cell_mat_m,
                construction_instructions,
                parameters,
                index_vec_h,
                computation_dict
            )
            f0j_core = np.array(f0j_core)
        else:
            f0j_core = np.array(calc_f0j_core(
                cell_mat_m,
                construction_instructions,
                parameters,
                index_vec_h,

                symm_mats_vecs,
                computation_dict
            ))
        if core_io == 'save':
            with open(core_file, 'wb') as fo:
                pickle.dump(f0j_core, fo)
            print('  saved core atomic form factors to disk')
        #f0j_core += f_dash[:, None]

    elif core == 'combine':
        pass
        #f0j_core = None
    else:
        raise NotImplementedError('Choose either constant or combine for core description')
        
    print('Calculating density')

    f0j = calc_f0j(
        cell_mat_m,           
        construction_instructions,
        parameters,
        index_vec_h,
        symm_mats_vecs,
        computation_dict,
        False,
        core == 'constant'
    )
    if tsc_core_path is None and core != 'combine':
        f0j = f0j + f0j_core[None, :, :]

    f0j2tsc(tsc_path, f0j, construction_instructions, symm_mats_vecs, index_vec_h, False)

    if tsc_core_path is not None:
        f0j_core_out = np.ones_like(f0j) * f0j_core[None,:,:]
        f0j2tsc(tsc_core_path, f0j_core_out, construction_instructions, symm_mats_vecs, index_vec_h, False)

    print('Calculation finished')