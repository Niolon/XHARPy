Here is the list of available examples, with connected keywords. All examples are written as
jupyter notebooks. So to run them, go to the respective folder and type ``jupyter lab``. 
If you are unfamiliar with jupyter, please consult one of the myriad tutorials on 
Youtube or the internet in general.

Calculation with GPAW 
---------------------

L-Alanin with GPAW
******************
- Folder: L-Alanin_gpaw
- Original data doi: `10.1021/j100315a022 <https://doi.org/10.1021/j100315a022>`_
- Program: GPAW
- SCAN calculation
- Using MPI
- Constant Core calculation

Urea with GPAW
**************
- Folder: Urea_gpaw
- Original data doi: `10.1107/S0108767304015120 <https://doi.org/10.1107/S0108767304015120>`_
- Program: GPAW
- SCAN calculation
- Using MPI
- Constant Core calculation
- Multi-Step refinement
- Save core density
- Gram-Charlier Refinement
- Atom table from refined parameters


Xylitol with GPAW
*****************
- Folder: Xylitol_gpaw_extinction
- Original data doi: `10.1107/S0108767304018306 <https://doi.org/10.1107/S0108767304018306>`_
- Program: GPAW
- SCAN calculation
- Using MPI
- Constant Core calculation
- SHELXL-style extinction correction


CpNa with GPAW
**************
- Folder: CpNa_gpaw_special_position
- Original data doi: `10.1002/anie.201304498 <https://doi.org/10.1002/anie.201304498>`_
- Program: GPAW
- RPBE calculation
- Constant core calculation
- Read xd.hkl
- Disorder on special position / skip atoms

Hexaqua magnesium hydrogen maleate with GPAW
********************************************
- Folder: HMa_Mg_gpaw
- Original data doi: `10.1039/D0CE00378F <https://doi.org/10.1039/D0CE00378F>`_
- Program: GPAW
- SCAN calculation
- Using MPI
- Constant Core calculation

8-Hydroxy quinoline hydrogen maleate with GPAW
**********************************************
- Folder: HMa_8HQ_gpaw
- Original data doi: `10.1039/D0CE00378F <https://doi.org/10.1039/D0CE00378F>`_
- Program: GPAW
- SCAN calculation
- Using MPI
- Constant Core calculation


Calculation with Quantum Espresso
---------------------------------

L-Alanin with Quantum Espresso
******************************

- Folder: L-Alanin\_qe
- Original data doi: `10.1021/j100315a022 <https://doi.org/10.1021/j100315a022>`_
- Program: Quantum Espresso
- PBE calculation
- Constant core calculation


Urea with Quantum Espresso
**************************

- Folder: Urea\_qe
- Original data doi: `10.1107/S0108767304015120 <https://doi.org/10.1107/S0108767304015120>`_
- Program: Quantum Espresso
- B3LYP calculation from PBE files
- Core scaling

Calculation with NoSpherA2/ORCA
-------------------------------

L-Alanin with NoSpherA2/Orca
********************************************

- Folder: L-Alanin\_nosphera2\_orca
- Original data doi: `10.1107/S0108767304015120 <https://doi.org/10.1107/S0108767304015120>`_
- Program: NoSpherA2/Orca
- B3LYP calculation
- 8 Angstroem cluster charges



Calculation with the Independent Atom model
-------------------------------------------

L-Alanin with the Independent Atom Model
****************************************
- Folder: L-Alanin_iam
- Original data doi: `10.1021/j100315a022 <https://doi.org/10.1021/j100315a022>`_
- Independent Atom Model
- Isotropic Hydrogen Refinement

L-Alanin with the Independent Atom Model and Hydrogen Constraints
*****************************************************************
- Folder: L-Alanin_iam_HConstraints
- Original data doi: `10.1021/j100315a022 <https://doi.org/10.1021/j100315a022>`_
- Independent Atom Model
- Isotropic Hydrogen Refinement
- Position constraints for sp3 hydrogen atoms
- U(equiv) constraints