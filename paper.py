from .io import ciflike_to_dict
from .conversion import ucif2ucart, cell_constants_to_M
from scipy.optimize import minimize
import numpy as np
import jax
import jax.numpy as jnp
import warnings
import pandas as pd
from cycler import cycler

bench_colors = cycler(color=['#011671', '#ea5900', '#b10f2e', '#357266', '#e8c547', '#437f97',
                             '#8697e0', '#ffb485', '#f58ea1', '#53b2a0', '#e8d99e', '#93bccc'])


def corr_uij_tric(parameters, uij_neut):
    scale = parameters[0]
    add = parameters[1:]
    return scale * uij_neut + add

def corr_uij_mono(parameters, uij_neut):
    scale = parameters[0]
    add = jnp.zeros(6)
    add = jax.ops.index_add(add, jnp.array([0, 1, 2]), parameters[1:4])
    add = jax.ops.index_add(add, 4, parameters[4])
    return scale * uij_neut + add

def corr_uij_ortho(parameters, uij_neut):
    scale = parameters[0]
    add = jnp.zeros(6)
    add = jax.ops.index_add(add, jnp.array([0, 1, 2]), parameters[1:])
    return scale * uij_neut + add

def corr_uij_tetra(parameters, uij_neut):
    scale = parameters[0]
    add = jnp.zeros(6)
    add = jax.ops.index_add(add, jnp.array([0, 1]), parameters[1])
    add = jax.ops.index_add(add, 2, parameters[2])
    return scale * uij_neut + add

def corr_uij_hexa(parameters, uij_neut):
    scale = parameters[0]
    add = jnp.zeros(6)
    add = jax.ops.index_add(add, jnp.array([0, 1]), parameters[1])
    add = jax.ops.index_add(add, 2, parameters[2])
    add = jax.ops.index_add(add, 5, parameters[1] / 2)
    return scale * uij_neut + add

def corr_uij_cubic(parameters, uij_neut):
    scale = parameters[0]
    add = jnp.zeros(6)
    add = jax.ops.index_add(add, jnp.array([0, 1, 2]), parameters[1])
    return scale * uij_neut + add

def corr_uij_tric_nosc(parameters, uij_neut):
    add = parameters
    return uij_neut + add

def corr_uij_mono_nosc(parameters, uij_neut):
    add = jnp.zeros(6)
    add = jax.ops.index_add(add, jnp.array([0, 1, 2]), parameters[:3])
    add = jax.ops.index_add(add, 4, parameters[3])
    return uij_neut + add

def corr_uij_ortho_nosc(parameters, uij_neut):
    add = jnp.zeros(6)
    add = jax.ops.index_add(add, jnp.array([0, 1, 2]), parameters)
    return uij_neut + add

def corr_uij_tetra_nosc(parameters, uij_neut):
    add = jnp.zeros(6)
    add = jax.ops.index_add(add, jnp.array([0, 1]), parameters[0])
    add = jax.ops.index_add(add, 2, parameters[1])
    return uij_neut + add

def corr_uij_hexa_nosc(parameters, uij_neut):
    add = jnp.zeros(6)
    add = jax.ops.index_add(add, jnp.array([0, 1]), parameters[0])
    add = jax.ops.index_add(add, 2, parameters[1])
    add = jax.ops.index_add(add, 5, parameters[0] / 2)
    return uij_neut + add

def corr_uij_cubic_nosc(parameters, uij_neut):
    add = jnp.zeros(6)
    add = jax.ops.index_add(add, jnp.array([0, 1, 2]), parameters[0])
    return uij_neut + add

def gen_lsq(corr_func, weights):
    def lsq(parameters, uij, uij_neut):
        return (jnp.sum(weights * (corr_func(parameters, uij_neut) - uij)**2) + 0 * (1 - parameters[0])**2) / np.sum(weights)
    return lsq

func_start = {
    'triclinic': (corr_uij_tric, corr_uij_tric_nosc, jnp.array([1] + [0] * 6)),
    'monoclinic': (corr_uij_mono, corr_uij_mono_nosc, jnp.array([1] + [0] * 4)),
    'orthorhombic': (corr_uij_ortho, corr_uij_ortho_nosc, jnp.array([1] + [0] * 3)),
    'tetragonal': (corr_uij_tetra, corr_uij_tetra_nosc, jnp.array([1] + [0] * 2)),
    'rhombohedral': (corr_uij_tric, corr_uij_tric_nosc, jnp.array([1] + [0] * 6)),
    'hexagonal': (corr_uij_hexa, corr_uij_hexa_nosc, jnp.array([1] + [0] * 2)),
    'cubic': (corr_uij_cubic, corr_uij_cubic_nosc, jnp.array([1] + [0] * 1))
}

def calc_s12(mat_u1, mat_u2):
    mat_u1_inv = np.linalg.inv(mat_u1)
    mat_u2_inv = np.linalg.inv(mat_u2)
    numerator = 2**(3.0/2.0) * np.linalg.det(mat_u1_inv @ mat_u2_inv)**0.25
    denominator = np.linalg.det(mat_u1_inv + mat_u2_inv)**0.5
    return 100 * (1 - numerator / denominator)

def calculate_agreement(har_path, har_key, fcf_path,  neut_path, neut_key, rename_dict={}):
    cell_keys = ['cell_length_a', 'cell_length_b', 'cell_length_c', 'cell_angle_alpha', 'cell_angle_beta', 'cell_angle_gamma']
    uij_keys = ['U_11', 'U_22', 'U_33', 'U_23', 'U_13','U_12']
    uij_std_keys = [label + '_std' for label in uij_keys]
    collect = {}
    
    cif_neut = ciflike_to_dict(neut_path)[neut_key]
    atom_table_neut = next(loop for loop in cif_neut['loops'] if 'atom_site_fract_x' in loop.columns)
    atom_table_neut.columns = [name[10:] for name in atom_table_neut.columns]
    hydrogen_atoms = list(atom_table_neut.loc[atom_table_neut['type_symbol'] == 'H', 'label'])
    non_hydrogen_atoms = list(atom_table_neut.loc[atom_table_neut['type_symbol'] != 'H', 'label'])
    uij_table_neut = next(loop for loop in cif_neut['loops'] if 'atom_site_aniso_U_11' in loop.columns)
    uij_table_neut.columns = [name[16:] for name in uij_table_neut.columns]
    bond_table_neut = next(loop for loop in cif_neut['loops'] if 'geom_bond_distance' in loop.columns)
    bond_table_neut.columns = [name[10:] for name in bond_table_neut.columns]
    h_bond_table_neut = bond_table_neut[bond_table_neut['atom_site_label_1'].isin(hydrogen_atoms)].copy()
    h_bond_table_neut = h_bond_table_neut.rename(columns={'atom_site_label_1': 'atom_site_label_2', 'atom_site_label_2': 'atom_site_label_1'})
    h_bond_table_neut = h_bond_table_neut.append(bond_table_neut[bond_table_neut['atom_site_label_2'].isin(hydrogen_atoms)])
    bond_pairs = [(el1, el2) for el1, el2 in zip(h_bond_table_neut['atom_site_label_1'],
                                                 h_bond_table_neut['atom_site_label_2'])]
    cell_neut = np.array([cif_neut[key] for key in cell_keys])
    cell_neut_std = np.array([cif_neut.get(key + '_std', 0.0) for key in cell_keys])
    h_indexes_neut = [list(uij_table_neut['label'].values).index(atom) for atom in hydrogen_atoms]
    non_h_indexes_neut = [list(uij_table_neut['label'].values).index(atom) for atom in non_hydrogen_atoms]
    try:
        lattice = cif_neut['space_group_crystal_system']
    except:
        lattice = cif_neut['symmetry_cell_setting']
    cif = ciflike_to_dict(har_path, har_key)

    atom_table = next(loop for loop in cif['loops'] if 'atom_site_fract_x' in loop.columns)
    atom_table.columns = [name[10:] for name in atom_table.columns]
    atom_table['label'] = [rename_dict.get(label, label) for label in atom_table['label']]
    
    uij_table = next(loop for loop in cif['loops'] if 'atom_site_aniso_U_11' in loop.columns)
    uij_table.columns = [name[16:] for name in uij_table.columns]
    uij_table['label'] = [rename_dict.get(label, label) for label in uij_table['label']]
    
    bond_table = next(loop for loop in cif['loops'] if 'geom_bond_distance' in loop.columns)
    bond_table.columns = [name[10:] for name in bond_table.columns]
    bond_table['atom_site_label_1'] = [rename_dict.get(label, label) for label in bond_table['atom_site_label_1']]
    bond_table['atom_site_label_2'] = [rename_dict.get(label, label) for label in bond_table['atom_site_label_2']]
    
    h_bond_table = bond_table[bond_table['atom_site_label_1'].isin(hydrogen_atoms)]
    h_bond_table = h_bond_table.rename(columns={'atom_site_label_1': 'atom_site_label_2', 'atom_site_label_2': 'atom_site_label_1'})
    h_bond_table = h_bond_table.append(bond_table[bond_table['atom_site_label_2'].isin(hydrogen_atoms)])

    cell = np.array([cif[key] for key in cell_keys])
    cell_std = np.array([cif.get(key + '_std', 0.0) for key in cell_keys])
    h_indexes = [list(uij_table['label'].values).index(atom) for atom in hydrogen_atoms]
    non_h_indexes = [list(uij_table['label'].values).index(atom) for atom in non_hydrogen_atoms]
    uij = jnp.array(uij_table.loc[non_h_indexes, uij_keys].values)
    uij_std = jnp.array(uij_table.loc[non_h_indexes, uij_std_keys].values)
    uij_neut_start = jnp.array(uij_table_neut.loc[non_h_indexes_neut, uij_keys].values)
    uij_neut_std = jnp.array(uij_table_neut.loc[non_h_indexes_neut, uij_std_keys].values)


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weights = np.array(1 / (uij_neut_std**2 + uij_std**2))
    weights = np.nan_to_num(weights, 0)
    weights[weights > 1e30] = 0
    corr_func, corr_func_nosc, start = func_start[lattice.lower()]
    
    lsq_func_nosc = gen_lsq(corr_func_nosc, jnp.array(weights))
    grad = jax.jit(jax.grad(lsq_func_nosc))
    x = minimize(lsq_func_nosc, jac=grad, x0=start[1:], args=(uij, uij_neut_start), tol=1e-20)
    parameters_nosc = x.x
    
    start = jax.ops.index_update(start, jax.ops.index[1:], parameters_nosc)
    
    lsq_func = gen_lsq(corr_func, jnp.array(weights))
    grad = jax.jit(jax.grad(lsq_func))
    x = minimize(lsq_func, jac=grad, x0=start, args=(uij, uij_neut_start), tol=1e-20)
    parameters = x.x
        
    merged_bonds = pd.merge(h_bond_table, h_bond_table_neut, on=['atom_site_label_1', 'atom_site_label_2'], suffixes=['_har', '_neut'])
    
    distances_har = merged_bonds['distance_har'].values
    distances_har_esd = merged_bonds['distance_std_har'].values
    distances_neut = merged_bonds['distance_neut'].values
    distances_neut_esd = merged_bonds['distance_std_neut'].values

    collect['<Delta r>'] = np.mean(distances_har - distances_neut)
    collect['ssd(<Delta r>)'] = np.std(distances_har - distances_neut)
    collect['<|Delta r|>'] = np.mean(np.abs(distances_har - distances_neut))
    collect['ssd(<|Delta r|>)'] = np.std(np.abs(distances_har - distances_neut))
    collect['wRMSD(Delta r)'] = np.sqrt(np.mean((distances_har - distances_neut)**2 / 
                                                (distances_har_esd**2 + distances_neut_esd**2)))
    collect['<rX/rN>'] = np.mean(distances_har / distances_neut)
    collect['ssd(<rX/rN>)'] = np.std(distances_har / distances_neut)
    if 'type_symbol' not in atom_table.columns:
        atom_table['type_symbol'] = [label[0] if label[1].isnumeric() else label[:2] for label in atom_table['label']]
    uij_merged = pd.merge(atom_table[['label', 'type_symbol']], uij_table, on='label')
    
    uij_merged = pd.merge(uij_merged, uij_table_neut, on='label', suffixes=['_har', '_neut_nosc'])
    uij_qdel = uij_table_neut.copy()
    uij_qdel[uij_keys] = corr_func(parameters, uij_qdel[uij_keys].values)
    uij_qdel.columns = [uij_qdel.columns[0]] + [name + '_neut_qdel' for name in uij_keys] + [name + '_std_neut_qdel' for name in uij_keys]
    uij_merged = pd.merge(uij_merged, uij_qdel, on='label', suffixes=[None, '_neut_qdel'])
    uij_onlydel = uij_table_neut.copy()
    uij_onlydel[uij_keys] = corr_func_nosc(parameters_nosc, uij_onlydel[uij_keys].values)
    uij_onlydel.columns = [uij_onlydel.columns[0]] + [name + '_neut_onlydel' for name in uij_keys] + [name + '_std_neut_onlydel' for name in uij_keys]
    uij_merged = pd.merge(uij_merged, uij_onlydel, on='label', suffixes=[None, '_neut_onlydel'])
    assert len(uij_merged) == len(uij_table_neut)
    uij_merged_h = uij_merged[uij_merged['type_symbol'] == 'H']
    cell_mat_m = cell_constants_to_M(*cell)
    cell_mat_m_neut = cell_constants_to_M(*cell_neut)


    for suffix in ['nosc', 'qdel', 'onlydel']:
        if suffix == 'qdel':
            collect[f'{suffix}:scale q'] = parameters[0]
            vals = corr_func(parameters, np.zeros(6))
            collect[f'{suffix}:scale Delta U11'] = float(vals[0])
            collect[f'{suffix}:scale Delta U22'] = float(vals[1])
            collect[f'{suffix}:scale Delta U33'] = float(vals[2])
            collect[f'{suffix}:scale Delta U23'] = float(vals[3])
            collect[f'{suffix}:scale Delta U13'] = float(vals[4])
            collect[f'{suffix}:scale Delta U12'] = float(vals[5])
            uij_merged[f'{suffix}:scale q'] = float(parameters[0])
            uij_merged[f'{suffix}:scale Delta U11'] = float(vals[0])
            uij_merged[f'{suffix}:scale Delta U22'] = float(vals[1])
            uij_merged[f'{suffix}:scale Delta U33'] = float(vals[2])
            uij_merged[f'{suffix}:scale Delta U23'] = float(vals[3])
            uij_merged[f'{suffix}:scale Delta U13'] = float(vals[4])
            uij_merged[f'{suffix}:scale Delta U12'] = float(vals[5])
            
        if suffix == 'onlydel':
            vals = corr_func_nosc(parameters_nosc, np.zeros(6))
            collect[f'{suffix}:scale Delta U11'] = float(vals[0])
            collect[f'{suffix}:scale Delta U22'] = float(vals[1])
            collect[f'{suffix}:scale Delta U33'] = float(vals[2])
            collect[f'{suffix}:scale Delta U23'] = float(vals[3])
            collect[f'{suffix}:scale Delta U13'] = float(vals[4])
            collect[f'{suffix}:scale Delta U12'] = float(vals[5])
            uij_merged[f'{suffix}:scale Delta U11'] = float(vals[0])
            uij_merged[f'{suffix}:scale Delta U22'] = float(vals[1])
            uij_merged[f'{suffix}:scale Delta U33'] = float(vals[2])
            uij_merged[f'{suffix}:scale Delta U23'] = float(vals[3])
            uij_merged[f'{suffix}:scale Delta U13'] = float(vals[4])
            uij_merged[f'{suffix}:scale Delta U12'] = float(vals[5])
            
        uij_har = uij_merged.loc[:, [key + '_har' for key in uij_keys]].values

        uij_neut = uij_merged.loc[:, [key + '_neut_' + suffix for key in uij_keys]].values

        uij_neut_esd = uij_merged[[name + '_std_neut_' + suffix for name in uij_keys]].values
        
        uij_merged[[f'{suffix}: Delta {u}' for u in ['U11', 'U22', 'U33', 'U23', 'U13', 'U12']]] = uij_har - uij_neut
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  
            uij_merged[[f'{suffix}: Delta {u}/sigmaN' for u in ['U11', 'U22', 'U33', 'U23', 'U13', 'U12']]] = (uij_har - uij_neut) / uij_neut_esd
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") 
            har_uij_mat = uij_har[:, [[0, 5, 4], [5, 1, 3], [4, 3, 2]]]
            har_uij_cart = ucif2ucart(cell_mat_m, har_uij_mat)
            neut_uij_mat = uij_neut[:, [[0, 5, 4], [5, 1, 3], [4, 3, 2]]]
            neut_uij_cart = ucif2ucart(cell_mat_m_neut, neut_uij_mat)
        
            uij_merged[f'{suffix}:S12'] = np.array([calc_s12(mat_u1, mat_u2) for mat_u1, mat_u2 in zip(har_uij_cart, neut_uij_cart)])
             
            
        compare_suffix = '_neut_' + suffix
        uij_har = uij_merged_h[[name + '_har' for name in uij_keys]].values
        uij_har_esd = uij_merged_h[[name + '_std_har' for name in uij_keys]].values

        uij_neut = uij_merged_h[[name + compare_suffix for name in uij_keys]].values
        uij_neut_esd = uij_merged_h[[name + '_std' + compare_suffix for name in uij_keys]].values

        har_uij_mat = uij_har[:, [[0, 5, 4], [5, 1, 3], [4, 3, 2]]]
        har_uij_cart = ucif2ucart(cell_mat_m, har_uij_mat)
        neut_uij_mat = uij_neut[:, [[0, 5, 4], [5, 1, 3], [4, 3, 2]]]
        neut_uij_cart = ucif2ucart(cell_mat_m_neut, neut_uij_mat)
        v_har = np.linalg.det(har_uij_cart)
        v_neut = np.linalg.det(neut_uij_cart)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        
            collect[f'{suffix}:<|Delta Uij|>'] = np.mean(np.abs(uij_har - uij_neut))
            collect[f'{suffix}:<Delta Uij>'] =  np.mean(uij_har - uij_neut)
            collect[f'{suffix}:wRMSD(Delta Uij)'] = np.sqrt(np.nanmean((uij_har - uij_neut)**2 / (uij_har_esd**2 + uij_neut_esd**2)))
            collect[f'{suffix}:S12'] = np.mean(np.array([calc_s12(mat_u1, mat_u2) for mat_u1, mat_u2 in zip(har_uij_cart, neut_uij_cart)]))
            collect[f'{suffix}:VX/VN'] =  np.mean(v_har / v_neut)
            collect[f'{suffix}:(VX-VN)/VN'] = np.mean((v_har - v_neut) / v_neut)
            collect[f'{suffix}:ssd(<|Delta Uij|>)'] = np.std(np.abs(uij_har - uij_neut))
            collect[f'{suffix}:ssd(<Delta Uij>)'] =  np.std(uij_har - uij_neut)
            collect[f'{suffix}:ssd(S12)'] = np.std(np.array([calc_s12(mat_u1, mat_u2) for mat_u1, mat_u2 in zip(har_uij_cart, neut_uij_cart)]))
            collect[f'{suffix}:ssd(VX/VN)'] =  np.std(v_har / v_neut)
            collect[f'{suffix}:ssd((VX-VN)/VN)'] = np.std((v_har - v_neut) / v_neut)
    fcf = next(loop for loop in ciflike_to_dict(fcf_path, har_key)['loops'] if 'refln_F_squared_meas' in loop.columns)
    intensity = fcf['refln_F_squared_meas'].values
    stderr = fcf['refln_F_squared_sigma'].values
    try:
        f_calc = fcf['refln_F_calc'].values
    except:
        f_calc = np.sqrt(fcf['refln_F_squared_calc'].values)
    f_obs = np.sign(intensity) * np.sqrt(np.abs(intensity))
    f_obs_safe = np.array(f_obs)
    f_obs_safe[f_obs_safe == 0] = 1e-9
    sigma_f_obs = 0.5 * stderr / np.abs(f_obs_safe)

    i_over_2sigma = intensity / stderr > 2
    collect['R(F)'] = np.sum(np.abs(np.abs(f_obs) - np.abs(f_calc))) / np.sum(np.abs(f_obs))
    collect['R(F, I>2sigma)'] = np.sum(np.abs(f_obs[i_over_2sigma] - np.abs(f_calc[i_over_2sigma]))) / np.sum(np.abs(f_obs[i_over_2sigma]))

    collect['R(F^2)'] = np.sum(np.abs(np.array(intensity) - np.abs(f_calc)**2)) / np.sum(np.array(intensity))

    collect['wR(F^2)'] = np.sqrt(np.sum(1/stderr**2 * (intensity -  np.abs(f_calc)**2)**2) / np.sum(1/stderr**2 * intensity**2))
    
    n_pars = cif['refine_ls_number_parameters']

    collect['GOF'] = np.sqrt(np.sum(1/stderr**2 * (intensity - np.abs(f_calc)**2)**2) / (len(intensity) - n_pars))
    merged_bonds = merged_bonds[[column for column in merged_bonds.columns if column.startswith('atom') or column.startswith('distance')]]
    return collect, merged_bonds, uij_merged

def plot_heatmap(ax, table, columns, rows, cm, cmap_type='diverging', esds=None):
    tmax = np.nanmax(table, axis=0)
    tmin = np.nanmin(table, axis=0)
    if cmap_type == 'diverging':
        larger = np.maximum.reduce((np.abs(tmax), np.abs(tmin)))
        show_table = table / larger
        im = ax.matshow(show_table, aspect='auto', cmap=cm, vmax=1, vmin=-1)
    else:
        show_table = (table - tmin) / (tmax - tmin)
        im = ax.matshow(show_table, aspect='auto', cmap=cm)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(rows)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(columns)
    ax.set_yticklabels(rows)
    if esds is not None:
        indexes = np.isfinite(1 / esds)
        orders = np.floor(np.log10(esds))
        smaller2 = np.full_like(esds, False, dtype=np.bool)
        smaller2[indexes] = np.array(esds[indexes]) * 10**(-orders[indexes]) < 1.95

        orders[smaller2] -= 1
        esd_table = np.round(esds / 10**orders, 0).astype(np.int64)

    # Loop over data dimensions and create text annotations.
    for i in range(len(rows)):
        for j in range(len(columns)):
            if cmap_type in ('diverging', 'brighttodark'):
                if np.abs(show_table[i, j]) < 0.5:
                    color = 'k'
                else:
                    color='w'
            else:
                if show_table[i, j] < 0.5:
                    color = 'w'
                else:
                    color='k'
            if not np.isfinite(table[i, j]):
                continue

            if esds is None:
                text = ax.text(j, i, f'{table[i, j]:4.2f}',
                               ha='center', va='center', color=color)
            elif orders[i, j] >= 0:
                text = ax.text(j, i, f'{int(np.round(table[i, j], int(-orders[i,j])))}({int(esd_table[i, j] * 10**orders[i, j])})',
                               ha='center', va='center', color=color)
            elif int(orders[i, j]) == -1 and esd_table[i, j] > 9:
                text = ax.text(j, i, f'{np.round(table[i, j], 1)}({esd_table[i, j] / 10})',
                               ha='center', va='center', color=color)
            
            else:
                template = f'{{val:{int(-orders[i, j] + 2)}.{int(-orders[i, j])}f}}({{esd_val}})'
                text = ax.text(j, i, template.format(**{'val':table[i, j], 'esd_val': esd_table[i, j]}),
                               ha='center', va='center', color=color)     


def figure_height(n_dataset):
    top = 0.293
    line = 0.185
    bottom = 0.255
    return top + n_dataset * line + bottom

def box_options(color, widths=0.35):
    return dict(
        boxprops = dict(color=color, facecolor=color),
        flierprops = dict(markerfacecolor=color, markeredgecolor='none', markersize=4),
        medianprops = dict(linewidth=1.0, color='#ffffff'),
        patch_artist=True,
        widths=widths
    )
