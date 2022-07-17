"""XHARPy (X-ray diffraction data Hirshfeld Atom Refinement in Python) is a 
library enabling the refinement against custom non-spherical atomic form
factors.
"""

from .io import(
    cif2data, lst2constraint_dict, write_cif, write_res, write_fcf,
    shelxl_hkl2pd, fcf2hkl_pd, add_density_entries_from_fcf, xd_hkl2pd,
    cif2tsc
) 

try:
    from .refine import refine
except:
    warnings.warn('refine module could not be imported, jax is probably missing')

from .structure.initialise import (
    create_construction_instructions, ConstrainedValues, UEquivConstraint,
    TorsionPositionConstraint, TrigonalPositionConstraint, 
    TetrahedralPositionConstraint
)

from .structure.construct import (
    create_atom_table
)
try:
    from .quality import calculate_quality_indicators
except:
    warnings.warn('quality module could not be imported, jax is probably missing')
