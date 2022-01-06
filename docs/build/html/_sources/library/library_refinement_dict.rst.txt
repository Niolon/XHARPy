Refinement Options
==================
These are the options that can be passed from to the ``refinement_dict``

f0j_source:
-----------

Source of the atomic form factors. The computation_dict 
will be passed on to this method. See the individual files in
f0j_sources for more information, by default 'gpaw'
Options: 'gpaw', 'iam', 'gpaw_mpi'
Some limitations: 'gpaw_spherical'

reload_step:   
------------
Starting with this step the computation will try to reuse the 
density, if this is implemented in the source, by default 1

core:
-----
If this is implemented in a f0j_source, it will integrate the 
frozen core density on a spherical grid and only use the valence
density for the updated atomic form factor options are 
'combine', which will not treat the core density separately,
'constant' which will integrate and add the core density without
scaling parameter and 'scale' which will refine a scaling 
parameter for the core density which might for systematic
deviations due to a coarse valence density grid (untested!)
By default 'constant'

extinction:
-----------
Use an extinction correction. Options:
 - 'none' means no extinction correction
 - 'shelxl' use the (empirical) formula used by SHELXL to correct to correct for extinction
 - 'secondary' see Giacovazzo et al. 'Fundamentals of Crystallography' (1992) p.97

By default 'none'

max_dist_recalc:
----------------
If the max difference in atomic positions is under this value in 
Angstroems, no new structure factors will be calculated, by
default 1e-6

max_iter:
---------
Maximum of refinement cycles if convergence not reached, by 
default: 100

min_iter:
---------
Minimum refinement cycles. The refinement will stop if the
wR2 increases if the current cycle is higher than min_iter,
by default 10

core_io:
--------
Expects a tuple where the first entry can be 'save', 'load', 'none'
which is the action that is taken with the core density. The 
second argument in the tuple is the filename, to which the core
density is saved to or loaded from