xHARPy
======


**xHARPY** (x-ray diffraction Hirshfeld Atom Refinement in Python) is a Python
library that enables refinement with custom atomic form factor calculations.

This is the first refinement library to implement the calculation of atomic form
factors from periodic PAW-DFT calculations. Independent atom model for comparison
and debugging is also available.

The library has been written with extensibility in mind. You can look at the 
f0j_sources folder for examples how to write a new atomic form factor source.

Refinement itself relies heavily on JAX for the automatic generation of 
gradients. This means that new features only have to be implemented in the loss
function. No explicit gradients are needed. 

If you have used the library in your research, please cite the paper where it
was originally published:

TODO: Insert paper here

This library was written during my PhD which was embedded in the research 
training group BENCh at the University of Göttingen, which is funded by 
the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - 389479699/GRK245

Prerequisites
-------------

The following packages in the following versions were used for development
 - python = 3.8.12
 - `numpy <https://numpy.org/>`_ = 1.20.3
 - `scipy <https://scipy.org/>`_ = 1.7.3
 - `pandas <https://pandas.pydata.org/>`_ = 1.3.5
 - `jax <https://jax.readthedocs.io/>`_ = 0.2.26

For atomic form factor calculation in GPAW
 - `gpaw <https://wiki.fysik.dtu.dk/gpaw/>`_ = 21.6.0
 - `ase <https://wiki.fysik.dtu.dk/ase/>`_ = 3.22.1 

For difference electron density calculation
 - `cctbx <https://cci.lbl.gov/cctbx_docs/index.html>`_ = 2021.11

This does not mean, that the library will not work with other versions. I tried
not to use the newest of features, but I do not have the means/time to test how
much older or newer the versions can be before things start to break.

Documentation
-------------

With the sphinx package 