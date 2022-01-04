Basic Use
=========

This is a commented step-by-step guide for the usage of xHARPy as a python
library. I try to also give some understanding what is happening. If you just
want results fast, I would suggest you adapt the L-alanin_gpaw example to your
structure. The examples folder is also very useful to look at applications after
you have a basic understanding of what is happening.

The imports here are split up according to the section, but of you would usually
still put all of them at the top of a .py or .ipynb file.

The usage of xHARPy as a to write refinement scripts can be split into four
distinct steps: Loading your data, setting refinement and computation options, 
refinement and writing your data to disk.

Loading your data
-----------------

For structures without atoms on special positions you just need a .cif and an 
.hkl file. If you have atom on special positions, which require restraints, the
easiest way is to adapt these from a SHELXL .lst file

So let us import the io functions

.. code-block:: python

    from xharpy import shelxl_hkl2pd, cif2data, lst2constraint_dict

As written before the last one is only strictly necessary if we have atoms 
on special positions. On the other hand it will also give the correct empty 
dictionary if there are no constaints resulting from special positions, so I 
usually just import it anyway.

Let us load the data from the cif-file using the cif2data function:

.. code-block:: python

    atom_table, cell, cell_esd, symm_mats_vecs, symm_strings, wavelength  = cif2data('iam.cif', 0)

Let us first talk about the arguments of the function: The first one is simply
the path to our cif file. The second argument can either be an integer or a 
string and can be used to select a specific data block within the .cif file.
We have used an integer, which the function will interpret as an index. So we 
are selecting the first dataset. If we give a string we select the dataset by 
name. So the string needs to match whatever is written behind ``data\_`` keyword
in the cif file.

From our function we have got a number of results, we need to further process.
Most importantly we have created an ``atom_table``, which contains all our atomic 
data in a pandas DataFrame. The columns are shortened versions of the naming in
the loop instruction. They are shortened by omitting the common beginning that
is unique to each table in a cif file. If values have an esd in the cif file
is is output to ``<column_name>_esd``. Missing values are represented by a numpy 
nan value. The ``atom_table`` can be manipulated to change values (especially the
adp type and parameters before we start our refinemen)

cell and cell_esd are just arrays containing the cell parameters and their 
estimated standard deviations. The wavelength is a float with the wavelength.
``symm_mats_vecs`` is a tuple containing the symmetry matrices and translation
vectors for the symmetry elements in the cif file. The ``symm_strings`` are
the corresponding strings.

Next let us load the reflection information with:

.. code-block:: python

    hkl = shelxl_hkl2pd('iam.hkl')

Which will again return a pandas DataFrame, which has the columns: h, k, l, 
intensity and int_esd, where the last is the estimated standard deviation of
the measured intensities.

Finally, for dealing with special position constraints we need a ``constraint_dict``.
As mentioned before we can generate this one from an .lst file. If you have 
hydrogen atoms on special positions, these need to be anisotropic in the run of 
SHELXL that has been used for the .lst generation. Usually, you would want to
fix the hydrogen (or all atoms) with AFIX 1 before you run that refinement.

.. code-block:: python

    constraint_dict = lst2constraint_dict('iam.lst')
    # constraint_dict = {} # This is also possible if there are no atoms on special positions

A constraint_dict is a nested Dict. The syntax is explained in a separate page
about :doc:`symmetry constraints <library_symm_con>`.

Setting options
---------------

In general there are two type of options represented by their own dictionaries. 
Options that concern the refinement routine and options that concern the 
computation routines that calculate the atomic form factors.

A basic example for a ``refinement_dict`` would look like this:

.. code-block:: python

    refinement_dict = {
        'f0j_source': 'gpaw', # GPAW with single-core
        #'f0j_source': 'gpaw_mpi', # GPAW with multi-core
        'core': 'constant', # treatment of the core density
        'extinction': 'none', # Refinement of extinction
        'reload_step': 1, # step where the density is reloaded from the save_file
        'save_file': 'gpaw_result.gpw' # File where the DFT result is stored
    }

You might notice that three of the options concern the computation of the
atomic form factors. However, these are still used in the refinement routine.

