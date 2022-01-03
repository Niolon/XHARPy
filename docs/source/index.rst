.. xHARPy documentation master file, created by
   sphinx-quickstart on Thu Dec 30 11:12:58 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to xHARPy's documentation!
==================================

**xHARPY** (x-ray diffraction Hirshfeld Atom Refinement in Python) is a Python library that enables refinement with custom atomic form factor calculations.
Current implemented and tested sources for atomic form factors are Hirshfeld Atom Refinement from periodic DFT calculations and the Independent Atom Model.

Refinement itself relies heavily on JAX for the automatic generation of gradients. This means that new features only have to be implemented in the loss
function. No explicit gradients are needed. 

.. note::

   This project is still at the beginning of its development. Things might fundamentally change with future versions.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   commandline
   modules
   xharpy 
   xharpy.f0j_sources
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
