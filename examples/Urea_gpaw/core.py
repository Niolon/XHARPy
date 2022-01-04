import pickle
from ase.spacegroup import crystal
from scipy.interpolate import interp1d
from scipy.integrate import simps
from ase.parallel import parprint
import gpaw
import numpy as np

from ase.units import Bohr

def f_core_from_spline(spline, g_k, k=13):
    r_max = spline.get_cutoff()
    r = np.zeros(2**k + 1)
    r[1:] = np.exp(-1 * np.linspace(1.25 * k, 0.0 , 2**k)) * r_max
    #r[0] = 0
    gr = r[None,:] * g_k[:,None]
    j0 = np.zeros_like(gr)
    j0[gr != 0] = np.sin(2 * np.pi * gr[gr != 0]) / (2 * np.pi * gr[gr != 0])
    j0[gr == 0] = 1
    y00_factor = 0.5 * np.pi**(-0.5)
    int_me = 4 * np.pi * r**2  * spline.map(r) * j0
    return simps(int_me, x=r) * y00_factor

with open('core_values.pic', 'rb') as fo:
    value_dict = pickle.load(fo)
atoms = crystal(symbols=value_dict['symm_symbols'],
                basis=value_dict['symm_positions'] % 1,
                cell=value_dict['cell_mat_m'].T)
calc = gpaw.GPAW(**value_dict['computation_dict'])
atoms.set_calculator(calc)
calc.initialize(atoms)
cell_inv = np.linalg.inv(atoms.cell.T).T
g_k3 = np.einsum('xy, zy -> zx', cell_inv, value_dict['index_vec_h'])
g_ks = np.linalg.norm(g_k3, axis=-1)
splines = {setup.symbol: setup.get_partial_waves()[:4] for setup in calc.density.setups}

f0j_core = {}
n_steps = 100
n_per_step = 50

for name, (_, _, nc, _) in list(splines.items()):
    if name in list(f0j_core.keys()):
        continue
    #if name == 'H':
    #    f0j_core[name] = np.zeros_like(g_ks)
    if len(g_ks) > n_steps * n_per_step:
        print(f'  calculating the core structure factor by spline for {name}')
        g_max = g_ks.max() * Bohr + 0.1
        #x_inv = np.linspace(-0.5, g_max, n_steps * n_per_step)
        k = np.log(n_steps * n_per_step)
        x_inv = np.exp(-1 * np.linspace(1.25 * k, 0.0, n_steps * n_per_step)) * g_max 
        x_inv[0] = 0
        f0j = np.zeros(n_steps * n_per_step)
        for index in range(n_steps):
            f0j[index * n_per_step:(index + 1) * n_per_step] = f_core_from_spline(nc, x_inv[index * n_per_step:(index + 1) * n_per_step], k=19) 
        f0j_core[name] = interp1d(x_inv, f0j, kind='cubic')(g_ks * Bohr)
    else:
        print(f'  calculating the core structure factor for {name}')
        f0j = np.zeros(len(g_ks))
        for index in range(n_per_step, len(g_ks) + n_per_step, n_per_step):
            start_index = index - n_per_step
            if index < len(g_ks):
                end_index = index
            else:
                end_index = len(g_ks)
            f0j[start_index:end_index] = f_core_from_spline(nc, g_ks[start_index:end_index] * Bohr, k=19)
        f0j_core[name] = f0j
del(calc)
del(atoms)

with open('core_r_values.pic', 'wb') as fo:
    pickle.dump(np.array([f0j_core[symbol] for symbol in value_dict['element_symbols']]), fo)
