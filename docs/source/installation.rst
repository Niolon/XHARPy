Installation
============

Packages
--------
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

Windows
-------
As the library relies on GPAW there is no direct straightforward way to 
install on Microsoft Windows Python installations. However, the library 
has been tested on the Linux Subsystem For Windows. So you can install that very
handy program from Microsoft and simply follow the Linux installation
afterwards.

Linux
-----
The easiest way to obtain all the necessary package is within a conda
environment. After installation of Anaconda_ create and activate a new
environment using

   conda create -n xharpy

   conda activate xharpy

You can now start by installing python in that environment

   conda install -c conda-forge python=3.8

Followed by installing the necessary packages

   conda install -c conda-forge jax=0.2.26 numpy=1.20.3 scipy=1.7.3 pandas=1.3.5

Currently xHARPy has no working possibility for calculating difference 
electron densities on its own. For this purpose we need the cctbx library. 
The library will however run without cctbx, with some features unavailable.

   conda install -c conda-forge cctbx=2021.11

If you want to use gpaw as source for the atomic form factors (recommended)

   conda install -c conda-forge ase=3.22.1 gpaw=21.6.0

Finally, the examples are written as jupyter notebooks. It also has proven 
to be good practice to write the refinements in jupyter, as it is easy to
further analyse the results. This is however not necessary.

   conda install -c conda-forge jupyter jupyterlab

Using other functionals in GPAW
-------------------------------
If installed via conda GPAW does bring the PAW setups for some functionals. If 
you wand to use metaGGA functionals further action is also not necessary, as
these will use the PBE setups. For usage of GGA or LDA functionals not included
follow these steps.

 (1) Go to your GPAW path by typing into the console:
    
    cd $GPAW_SETUP_PATH

 (2) For all the main and transition group elements with the functional type *xc* in:
   
   gpaw-setup -f *xc* H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi

For more details consult the `gpaw documentation <https://wiki.fysik.dtu.dk/gpaw/>`_


.. _Anaconda: https://www.anaconda.com/products/individual

