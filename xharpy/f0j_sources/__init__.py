"""These modules provide the possibility to generate atomic form factors for the
use in xHARPy. 

Usage
-----
Each of the modules has to implement three functions:

calc_f0j
    Calculates the atomic form factors that have to be recalculated during the 
    refinement

calc_f0j_core
    Calculates the frozen core density on a spherical grid. This means it is
    only calculated once for each refinement. The core density can therefore
    be integrated on much smaller steps close to the core position

generate_cif_output
    Generates the output for the cif file that represents how the atomic 
    form factors were obtained

Development
-----------
If you want to write your own new way of calculating atomic form factors you
can write a new module within the f0j_sources folder.

In addition to the the implementation of the three functions they need to be
added to the core.refine function and to the io.write_cif functions with the
same selection key in the refinement_dict['f0j_source']


"""