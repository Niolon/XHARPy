"""XHARPy (X-ray diffraction data Hirshfeld Atom Refinement in Python) is a 
library enabling the refinement against custom non-spherical atomic form
factors.
"""

from .io import (
    cif2data, lst2constraint_dict, write_cif, write_res, write_fcf,
    shelxl_hkl2pd, fcf2hkl_pd, add_density_entries_from_fcf, xd_hkl2pd,
    cif2tsc
) 
import warnings

# Define base __all__ list with functions that are always available
__all__ = [
    # From io module
    'cif2data', 
    'lst2constraint_dict', 
    'write_cif', 
    'write_res', 
    'write_fcf',
    'shelxl_hkl2pd', 
    'fcf2hkl_pd', 
    'add_density_entries_from_fcf', 
    'xd_hkl2pd',
    'cif2tsc',
    
    # From structure.initialise
    'create_construction_instructions', 
    'ConstrainedValues', 
    'UEquivConstraint',
    'TorsionPositionConstraint', 
    'TrigonalPositionConstraint', 
    'TetrahedralPositionConstraint',
    
    # From structure.construct
    'create_atom_table',
]

# Conditionally add refine module
try:
    from .refine import refine #noqa: F401
    __all__.append('refine')
except ModuleNotFoundError:
    warnings.warn('refine module could not be imported, jax is probably missing')

from .structure.initialise import (
    create_construction_instructions, ConstrainedValues, UEquivConstraint,
    TorsionPositionConstraint, TrigonalPositionConstraint, 
    TetrahedralPositionConstraint
)

from .structure.construct import (
    create_atom_table
)

# Conditionally add quality module
try:
    from .quality import calculate_quality_indicators #noqa: F401
    __all__.append('calculate_quality_indicators')
except ModuleNotFoundError:
    warnings.warn('quality module could not be imported, jax is probably missing')