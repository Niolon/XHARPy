.. XHARPy documentation master file, created by
   sphinx-quickstart on Thu Dec 30 11:12:58 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to XHARPy's documentation!
==================================

**XHARPy** (X-ray diffraction data Hirshfeld Atom Refinement in Python) is a Python
library that enables refinement with custom atomic form factor calculations.

This is the first refinement library to implement the calculation of atomic form
factors from periodic PAW-DFT calculations. Currently the following sources for
atomic form factors are available:

- Periodic PAW with GPAW
- Periodic PAW with Quantum Espresso
- Independent Atom model 
- tsc files written by other programs

The library has been written with extensibility in mind. You can look at the 
f0j_sources folder for examples how to write a new atomic form factor source.

Refinement itself relies heavily on JAX for the automatic generation of 
gradients. This means that new features only have to be implemented in the loss
function. No explicit gradients are needed. 

If you have used the library in your research, please cite the paper where it
was originally published:

TODO: Insert paper here

The source code is available at `https://github.com/Niolon/XHARPy <https://github.com/Niolon/XHARPy>`_

Creation of this library was only possible due the generous support of the 
`Research Training Group BENCh <bench.uni-goettingen.de>`_ at the University 
of Göttingen, which is funded by 
the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - 389479699/GRK245

.. note::

   This project is still at the beginning of its development. Things might
   fundamentally change with future versions.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   commandline
   library/library_index
   examples
   modules
   xharpy 
   xharpy.f0j_sources

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
