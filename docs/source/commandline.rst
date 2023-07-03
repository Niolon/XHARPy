
Command Line Interface
======================

There is a basic command line interface for using XHARPy with GPAW starting from 
a SHELXL refinement. After you have installed xharpy, it is available from 
terminal by typing (make sure you are in the correct conda environment):

.. code-block:: console

    python -m xharpy.cli_refine

You can either give the needed value with the prompts in the command line
interface or as arguments when you call the cli. For a list of available options
type:

.. code-block:: console

    python -m xharpy.cli_refine --help

The command line interface is only meant as a quick tool to try out the 
refinement. For anything where a more complete access to the options of GPAW are
needed look at the scripting and/or the examples section.