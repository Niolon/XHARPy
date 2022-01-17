Basic Use
=========

This is a commented step-by-step guide for the usage of XHARPy as a python
library. I try to also give some understanding what is happening. If you just
want results fast, I would suggest you adapt the L-alanin_gpaw example to your
structure. The examples folder is also very useful to look at applications after
you have a basic understanding of what is happening.

The imports here are split up according to the section, but of you would usually
still put all of them at the top of a .py or .ipynb file.

The usage of XHARPy as a to write refinement scripts can be split into four
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
        'reload_step': 1, # step where the density is reloaded from the save_file, 1 means first step AFTER initialisation
    }

You might notice that two of the options concern the computation of the
atomic form factors. The ``f0j_source`` is used to actually select the 
implementation of the atomic form factor calculation within the refinement 
routine. The implementations are also unaware of the step in the refinement. 
The refinement itself triggers the reloading of a precalculated density.
We want to start from a new density, but after initialisation we want to reload
previous calculation to speed things up. We also want to calculate core density
on a separate spherical grid, as they have sharp maxima at the core positions. 
This might not be well described on the rectangular grid we use for the valence
density. This also means Hirshfeld partitioning will not affect the core density.
There are more options for the ``refinement_dict``, which are explained on a
:doc:`separate page <library_refinement_dict>`.

Next we need to define the options for the atomic form factor calculation. these
are directly passed on to the routines that we loaded with the f0j_source. An 
example the selected GPAW source and a molecular structure might look like this:

.. code-block:: python

    computation_dict = {
        # options for the XHARPy implementation
        'save_file': 'gpaw_result.gpw', # Where are results saved and loaded
        'gridinterpolation': 4, # density interpolation to use for Hirshfeld and FFT

        # options that are passed on to the gpaw calculator
        'xc': 'SCAN', # Functional
        'txt': 'gpaw.txt', # Text output for GPAW
        'h': 0.175, # Grid spacing for wavefunction calculation
        'convergence':{'density': 1e-7}, # Higher convergence for density calculation
        'symmetry': {'symmorphic': False}, # Also search for symmetry involving translation
        'nbands': -2 # Number of calculated bands = n(occ) + 2
    }

As you can see the function of the GPAW source will read the options that are 
specific to the XHARPy GPAW plugin and remove it from the dictionary. All options 
that are not known will be passed on to the GPAW calculator without any further 
checks. Options for the calc_f0j function can be found in the specific docstrings or 
here in the xharpy.f0j_sources page. GPAW options can be found in the 
`GPAW documentation <https://wiki.fysik.dtu.dk/gpaw/documentation/basic.html>`_

Refinement
----------

For refinement we need to import two additional functions

.. code-block:: python

    from xharpy import create_construction_instructions, refine

As mentioned on the introduction XHARPy uses JAX to automatically generate
gradients. However, we want to have one object that can map an array of
parameters to the properties of the atoms within the unit cell. Because of the 
implementation in JAX, using just-in-time compiling, that object has to be
immutable. We get it and starting values for the parameters by calling the 
``create_construction_instructions`` function:

.. code-block:: python

    construction_instructions, parameters = create_construction_instructions(
        atom_table=atom_table,
        constraint_dict=constraint_dict,
        refinement_dict=refinement_dict
    )

As you see we also need to pass the constraint_dict from the first section, as 
well as our refinement_dict in order to reserve additional parameters for things
like extinction.

Finally, we can call the refine function, to do our actual refinement:

.. code-block:: Python

    parameters, var_cov_mat, information = refine(
        cell=cell, 
        symm_mats_vecs=symm_mats_vecs,
        hkl=hkl,
        construction_instructions=construction_instructions,
        parameters=parameters,
        wavelength=wavelength,
        refinement_dict=refinement_dict,
        computation_dict=computation_dict
    )

The refinement will always refine the scale factor first before the atomic 
parameters are refined.

We get back a refined set of parameters, the variance-covariance matrix and 
an additional dictionary that contains things that might be interesting (such as
starting and end time) and things that are needed for output (such as the atomic
form factor values or the shifts at the last step).

Writing data to disk
--------------------

Finally we want to export our structures. There are three kinds of files that we
can write at the moment, and four functions that we need to import
    
.. code-block:: python

    from xharpy import write_cif, write_res, write_fcf, add_density_entries_from_fcf

The *crystallographic information file* is a standard format for exchanging and
depositing crystallographic data. We can write such a file with:

.. code-block:: python

    write_cif(
        output_cif_path='xharpy.cif',
        cif_dataset='xharpy',
        shelx_cif_path='iam.cif',
        shelx_dataset=0,
        cell=cell,
        cell_esd=cell_esd,
        symm_mats_vecs=symm_mats_vecs,
        hkl=hkl,
        construction_instructions=construction_instructions,
        parameters=parameters,
        var_cov_mat=var_cov_mat,
        refinement_dict=refinement_dict,
        computation_dict=computation_dict,
        information=information
    )

You might notice that we need an original cif file (the library was developed
wth SHELXL) to generate the new cif file. The reason is that the write-routine
does currently not calculate all values by itself. Additional values such as 
crystal size can also be added to the original cif file and will be then copied 
to the new one.

Fcf files can be written as fcf mode 4 or 6 with the two commands:

.. code-block:: python

    write_fcf(
        fcf_path='xharpy.fcf',
        fcf_dataset='xharpy',
        fcf_mode=4,
        cell=cell,
        hkl=hkl,
        construction_instructions=construction_instructions,
        parameters=parameters,
        wavelength=wavelength,
        refinement_dict=refinement_dict,
        symm_strings=symm_strings,
        information=information,
    )

.. code-block:: python

    write_fcf(
        fcf_path='xharpy_6.fcf',
        fcf_dataset='xharpy_6',
        fcf_mode=6,
        cell=cell,
        hkl=hkl,
        construction_instructions=construction_instructions,
        parameters=parameters,
        wavelength=wavelength,
        refinement_dict=refinement_dict,
        symm_strings=symm_strings,
        information=information,
    )

Both outputs will correct for extinction, but only fcf6 will correct the
observed reflections for dispersion effects. If you want to access the corrected
values for validation. Both functions return a pandas DataFrame.

XHARPy currently has no means of evaluating the difference electron density by 
itself. For this reason we need to use an additional function with a cctbx module
to add the missing entries to the cif file. 

.. code-block:: python

    add_density_entries_from_fcf('xharpy.cif', 'xharpy_6.fcf')

For visualisation of the structure and the difference electron density is is
also helpful to write a SHELXL .res file. This can be done by: 

.. code-block:: python

    write_res(
        out_res_path='xharpy_6.res',
        in_res_path='iam.lst',
        cell=cell,
        cell_esd=cell_esd,
        construction_instructions=construction_instructions,
        parameters=parameters,
        wavelength=wavelength
    )

Again we need a template res or lst file. Currently XHARPy has no way to divide
symmetry cards into those generated by a lattice centring or inversion symmetry 
and those generated by other symmetry elements, which would be necessary for 
writing these files on its own.
