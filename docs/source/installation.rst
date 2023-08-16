Installation
============

Packages
--------
The following packages are required:
 - python \>= 3.8.12
 - `numpy <https://numpy.org/>`_ ≥ 1.20.3
 - `scipy <https://scipy.org/>`_ ≥ 1.7.3
 - `pandas <https://pandas.pydata.org/>`_ ≥ 1.3.5
 - `jax <https://jax.readthedocs.io/>`_ ≥ 0.2.26

For atomic form factor calculation in GPAW
 - `gpaw <https://wiki.fysik.dtu.dk/gpaw/>`_ ≥ 21.6.0
 - `ase <https://wiki.fysik.dtu.dk/ase/>`_ ≥ 3.22.1

For atomic form factor calculation in Quantum Espresso (experimental)
 - `qe <https://www.quantum-espresso.org/>`_ ≥ 7.0

For difference electron density calculation
 - `cctbx <https://cci.lbl.gov/cctbx_docs/index.html>`_ ≥ 2021.11

Windows
-------
As the library relies on GPAW and JAX there is no direct straightforward way to
install on Microsoft Windows Python installations. However, the library
has been tested on the
`Windows Subsystem For Linux <https://docs.microsoft.com/en-us/windows/wsl/install>`_.
This provides a full linux shell under windows and can be installed to run linux
programs on Microsoft Windows.

.. note::

   The very short instructions given here only work on a current version of Windows 10/11,
   which means newer than build 2004. For older versions refer to
   `Microsofts Manual installations steps <https://docs.microsoft.com/en-us/windows/wsl/install-manual>`_

Open a Windows PowerShell in Administrator mode. You can do this by typing PowerShell
into your Windows Menu, right clicking on the entry and select "Run as Administrator"

In this console you can now type:

.. code-block:: console

   wsl --install

Afterwards, you probably need to restart your computer. Now open a new PowerShell and
type ``wsl``. If you now now have a linux console, everything works as intended
and you can go to the Linux section.

Sometimes it has happened that the default ubuntu installation is not present and wsl does not work.
**Only in that case** you need to type:

.. code-block:: console

   wsl --install -d Ubuntu

To start the installation of Ubuntu. Afterwards, start the wsl by typing the wsl command.
You can now follow the Linux instructions from that shell.

You can go to you harddrive c: by typing.

.. code-block:: console

   cd /mnt/c


Linux
-----
The easiest way to obtain all the necessary package is within a conda
environment. So first you need to install Anaconda_.

The fastest way from console is installation with the following commands:

.. code-block:: console

   wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
   bash Anaconda3-2021.11-Linux-x86_64.sh
   # Accept licence Agreement and run conda init
   source ~/.bashrc
   rm Anaconda3-2021.11-Linux-x86_64.sh

Note that we are playing it save for the instructions.
Installation without the version numbers (*i.e.* using the newest version)
should also work, at least for some time.

In Anaconda create and activate a new environment using

.. code-block:: console

   conda create -n xharpy
   conda activate xharpy


If you want to get everything as quickly as possible just type:

.. code-block:: console
   conda install python=3.11 numpy scipy pandas jax cctbx ase gpaw qe

This will install the somewhat new versions which should work at the time of writing (16. August 2023).
If it does not please raise an issue on Github and try the older explicit versions and instructions given below.

You can download the XHARPy library from: `https://github.com/Niolon/XHARPy <https://github.com/Niolon/XHARPy>`_.
You need to add the folder where your xharpy *directory* is located to the
$PYTHONPATH. If you are unsure the directory should be one level above the one
where the ``__init__.py`` is located, so if you have downloaded the complete
repository it is the folder containing the docs, examples and xharpy folder.
As long as there is no installation routine you can do this by:

.. code-block:: console

   conda develop /path/to/xharpy


Detailed installation with tested version numbers
-------------------------------------------------
You can now start by installing python in that environment

.. code-block:: console

   conda install -c conda-forge python=3.8

Followed by installing the necessary packages

.. code-block:: console

   conda install -c conda-forge jax=0.2.26 numpy=1.20.3 scipy=1.7.3 pandas=1.3.5

This will create a .pth file in the site-packages of your conda environment,
which acts as the necessary link for conda/python.

Currently XHARPy has no working possibility for calculating difference
electron densities on its own. For this purpose we need the cctbx library.
The library will however run without cctbx, with some features unavailable.

.. code-block:: console

   conda install -c conda-forge cctbx=2021.11

If you want to use gpaw as source for the atomic form factors (recommended)

.. code-block:: console

   conda install -c conda-forge ase=3.22.1 gpaw=21.6.0

You can download the XHARPy library from: `https://github.com/Niolon/XHARPy <https://github.com/Niolon/XHARPy>`_.
You need to add the folder where your xharpy *directory* is located to the
$PYTHONPATH. If you are unsure the directory should be one level above the one
where the ``__init__.py`` is located, so if you have downloaded the complete
repository it is the folder containing the docs, examples and xharpy folder.
As long as there is no installation routine you can do this by:
If you also want to try out the atomic form factor calculation in Quantum
Espresso, you need to install the program. You can do this with conda.
You can also install this separately of course, as long as pw.x
and pp.x are directly callable.

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


