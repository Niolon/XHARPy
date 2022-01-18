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

For atomic form factor calculation in Quantum Espresso (experimental)
 - `qe <https://www.quantum-espresso.org/>`_ = 7.0

For difference electron density calculation
 - `cctbx <https://cci.lbl.gov/cctbx_docs/index.html>`_ = 2021.11

This does not mean, that the library will not work with other versions. I tried
not to use the newest of features, but I do not have the means/time to test how
much older or newer the versions can be before things start to break.

Windows
-------
As the library relies on GPAW there is no direct straightforward way to 
install on Microsoft Windows Python installations. However, the library 
has been tested on the
`Windows Subsystem For Linux <https://docs.microsoft.com/en-us/windows/wsl/install>`_.
This provides a full linux shell under windows and can be installed to run linux
programs on Microsoft Windows.

If your Windows version is rather old, you need to follow the separate instructions 
in the provided link above. Here I will give the short version, that is possible
in a recent Windows 10/11 version.

Open a Windows PowerShell in Administrator mode. You can do this by typing PowerShell 
into your Windows Menu, right clicking on the entry and select "Run as Administrator"

In this console you can now type:

.. code-block:: console

   wsl --install

Afterwards, you probably need to restart your computer. Now open a new PowerShell and 
type ``wsl``. If you now now have a linux console you can go to the Linux section.

Sometimes it has happened that no default ubuntu installation is not present and wsl does not work. 
**Only in that case** you need to type:

.. code-block:: console

   wsl --install -d Ubuntu

To start the installation of Ubuntu. Afterwards, start the wsl by typing the wsl command. 
You can now follow the Linux instructions from that shell.


Linux
-----
The easiest way to obtain all the necessary package is within a conda
environment. So first you need to install Anaconda_. 
Instruction on how to do that in console only (for example in a WSL) see `here <https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da>`_.

In Anaconda create and activate a new environment using

.. code-block:: console

   conda create -n xharpy
   conda activate xharpy

You can now start by installing python in that environment

.. code-block:: console

   conda install -c conda-forge python=3.8

Followed by installing the necessary packages

.. code-block:: console

   conda install -c conda-forge jax=0.2.26 numpy=1.20.3 scipy=1.7.3 pandas=1.3.5

You need to add the folder where your xharpy *directory* is located to the 
$PYTHONPATH. If you are unsure the directory should be one level above the one
where the ``__init__.py`` is located. As long as there is no installation
routine you can do this by

.. code-block:: console

   conda develop /path/to/xharpy

This will create a .pth file in the site-packages of your conda environment.

Currently XHARPy has no working possibility for calculating difference 
electron densities on its own. For this purpose we need the cctbx library. 
The library will however run without cctbx, with some features unavailable.

.. code-block:: console

   conda install -c conda-forge cctbx=2021.11

If you want to use gpaw as source for the atomic form factors (recommended)

.. code-block:: console

   conda install -c conda-forge ase=3.22.1 gpaw=21.6.0

If you also want to try out the atomic form factor calculation in Quantum
Espresso, you need to the program. You can do this with conda.
You can also install this separately of course, as long as pw.x 
and pp.x directly callable.

.. code-block:: console

   conda install -c conda-forge qe=7.0

Finally, the examples are written as jupyter notebooks. It also has proven 
to be good practice to write the refinements in jupyter, as it is easy to
further analyse the results. This is however not necessary.

.. code-block:: console

   conda install -c conda-forge jupyter jupyterlab


Using other functionals in GPAW
-------------------------------

If installed via conda GPAW does bring the PAW setups for some functionals. If 
you wand to use metaGGA functionals further action is also not necessary, as
these will use the PBE setups. For usage of GGA or LDA functionals not included
follow these steps.

 (1) Go to your GPAW path by typing into the console:
   .. code-block:: console

      cd $GPAW_SETUP_PATH

 (2) For all the main and transition group elements with the functional type *xc* in:
   .. code-block:: console
   
      gpaw-setup -f *xc* H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi


For more details consult the `gpaw documentation <https://wiki.fysik.dtu.dk/gpaw/>`_


.. _Anaconda: https://www.anaconda.com/products/individual


