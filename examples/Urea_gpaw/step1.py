from ase.spacegroup import crystal
from ase.parallel import parprint
import gpaw
import warnings
import pickle

with open('step1_values.pic', 'rb') as fo:
    value_dict = pickle.load(fo)

e_change = True
if value_dict['restart'] is None:
    atoms = crystal(**value_dict['kw_crystal'])
    calc = gpaw.GPAW(**value_dict['kw_gpaw'])
    atoms.set_calculator(calc)
    e1 = atoms.get_potential_energy()
else:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms, calc = gpaw.restart(value_dict['restart'], txt=value_dict['kw_gpaw']['txt'], xc=value_dict['kw_gpaw']['xc'])
            e1_0 = atoms.get_potential_energy()

            atoms.set_scaled_positions(value_dict['kw_crystal']['basis'])
            e1 = atoms.get_potential_energy()
            e_change = abs(e1_0 - e1) > 1e-20
    except:
        parprint('  failed to load the density from previous calculation. Starting from scratch')
        atoms = crystal(**value_dict['kw_crystal'])
        calc = gpaw.GPAW(**value_dict['kw_gpaw'])
        atoms.set_calculator(calc)
        e1 = atoms.get_potential_energy()

e1 = atoms.get_potential_energy()
if value_dict['save'] is None:
    value_dict['save'] = 'save.gpw'
if e_change:
    try:
        calc.write(value_dict['save'], mode='all')
    except:
        parprint('  Could not save to file')
