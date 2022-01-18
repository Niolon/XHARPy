Constraints
===========

Special Position Constraints
----------------------------

Atoms on special positions often need to be refined with a reduced set of
parameters, as the atomic properties themselves need to obey the symmetry 
elements of the special positions. This is done by introducing special
position constraints.

The easiest way to generate special position constraints is by using the 
``lst2constraint_dict`` function. If there is no .lst available you can also
write the constraits by hand using the ``ConstrainedValues`` namedTuples.
If you want to see a finished example where constraints are used for 
Gram-Charlier parameters look at Urea\_gpaw in the examples folder.

So in a first step we need to import the necessary namedTuple.

.. code-block:: python

    from xharpy  import ConstrainedValues

Let us say we have an atom O2 located on a two-fold axis with the special
position ``0.5 y 0.5``. This means the position is constrained. If we do not
have a pre-existing constraint_dict to modify, we would generate a new one.

.. code-block:: python

    constraint_dict = {
        'O2': {
            'xyz': ConstrainedValues(
                variable_indexes = (-1, 0, -1),
                multiplicators = (0.0, 1.0, 0.0),
                added_value = (0.5, 0.0, 0.5),
                special_position=True
            )
        }
    }

We will now go through the entries within the ``ConstrainedValues`` line by line:

  - variable_indexes: Indicates whether the entry does need a new variable or
    not. A value of -1 does mean there is no refined variable associated. All
    the other variables are counted up from zero. So the second variable will
    be attached to a new refined value.
  - multiplicators: The value of the variable will be multiplied with this
    value. Is only important if a value is used several times, e.g. if the 
    variable indexes were (-1, 0, 0) so both the second and the thirds value
    were generated from one parameter.
  - added value: After multiplication this value is added. The combination with
    variable index -1 and multiplicator 0.0 means that the value is not refined,
    but set to the given value
  - special_position. Boolean indicating, whether this Constraint comes from 
    a special position. Is currently only used for the occupancy output into 
    the cif file.

We are currently missing the uij and occupancy constraints. We can add these
with: 

.. code-block:: python

    constraint_dict['O2']['uij'] = ConstrainedValues(
        variable_indexes = (0, 1, 2, -1, 3, -1),
        multiplicators = (1.0, 1.0, 1.0, 0.0, 1.0, 0.0),
        added_value = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        special_position = True
    )

    constraint_dict['O2']['occ'] = ConstrainedValues(
        variable_indexes = (-1),
        multiplicators = (0.0),
        added_value = (0.5),
        special_position = True
    )

The individual variable_indexes and multiplicators can also consist of a tuple,
if all indexes in the tuple have been used before.
In this case the value at this position will be generated as a combination of 
the two parameters. The added value is still a single value in this case.

Finally, the ordering of the individual parameters is as follows:

===== =======================================================================================================
name  order
===== =======================================================================================================
xyz   x, y, z

uij   U11, U22, U33, U23, U13, U12

cijk  C111, C222, C333, C112, C122, C113, C133, C223, C233, C123

dijkl D1111, D2222, D3333, D1112, D1222, D1113, D1333, D2223, D2333, D1122, D1133, D2233, D1123, D1223, D1233

occ   occ
===== =======================================================================================================

Position and ADP constraints
----------------------------

In order to place hydrogen atoms three position constraints and one ADP 
constraint are available. There constraints themselves link the values to the 
variables of the connected atoms and refinement is done using gradients 
resulting from both atom, whose values were linked. This means adding
a constraint will slightly affect the parameters of the parent atom,
even if the effects should be low, as hydrogen atoms do not scatter much. 

U(equiv)-constraint
*******************

The application of a U(equiv) constraint is straightforward. Let us say we have
an atom H5 that is connected to a carbon atom C5, which is non-terminal.

.. code-block:: python

    constraint_dict['H5'] = {
        'uij': UEquivConstraint(
            bound_atom='C5',
            multiplicator=1.2 # C5 is non-terminal
            # multiplicator=1.5 # C5 is terminal
        )
    }

Single hydrogen sp3 atom
************************

We can generate a new hydrogen position by adding the direction of the three other
bonds connected to the sp3 atom and placing the hydrogen atom with a given 
distance in that direction. If we have already added the UEquivConstraint, we 
can simply add an expression for xyz to that dictionary.

.. code-block:: python

    constraint_dict['H5']['xyz'] = TetrahedralPositionConstraint(
        bound_atom_name='C5',        # name of bound atom 
        tetrahedron_atom1_name='C6', # name of first atom forming the tetrahedron
        tetrahedron_atom2_name='C4', # name of second atom forming the tetrahedron
        tetrahedron_atom3_name='C10',# name of third atom forming the tetrahedron
        distance=0.98                # interatomic distance
    )

Here C4, C6 and C10 are the three other atoms connected to C5, where H5 completes
the tetrahedron

Two hydrogen sp3 atoms
**********************

A different case would be two hydrogen atoms connected to a carbon atom. We can 
set this case with a TorsionPositionConstraint. Let C4 and C6 be the connected
carbon atoms, and H5A and H5B be the connected hydrogen atoms. We could
construct our constraints as follows:

.. code-block:: python

    constraint_dict['H5A']['xyz'] = TorsionPositionConstraint(
        bound_atom_name='C5',
        angle_atom_name='C4',
        torsion_atom_name='C6',
        distance=0.98,
        angle=109.47,
        torsion_angle_add=120,
        refine=False
    )

    constraint_dict['H5B']['xyz'] = TorsionPositionConstraint(
        bound_atom_name='C5',
        angle_atom_name='C4',
        torsion_atom_name='C6',
        distance=0.98,
        angle=109.47,
        torsion_angle_add=-120,
        refine=False
    )

Three hydrogen sp3 atoms
************************

Three atoms are also refined using a TorsionPositionConstraint. However, we can 
either use two bound atoms again or define the torsion angle along existing bonds
If we set ``refine=True`` the ``create_construction_instructions`` routine will
try to guess a good starting value. Parameters that are defined along the same 
three atoms will also be refined with one parameter for the Torsion angle. An 
example with refined torsion angle would look like this:

.. code-block:: python

    constraint_dict['H5A']['xyz'] = TorsionPositionConstraint(
        bound_atom_name='C5',
        angle_atom_name='C4',
        torsion_atom_name='C3',
        distance=0.98,
        angle=109.47,
        torsion_angle_add=0,
        refine=True
    )

    constraint_dict['H5B']['xyz'] = TorsionPositionConstraint(
        bound_atom_name='C5',
        angle_atom_name='C4',
        torsion_atom_name='C3',
        distance=0.98,
        angle=109.47,
        torsion_angle_add=120,
        refine=True
    )

    constraint_dict['H5C']['xyz'] = TorsionPositionConstraint(
        bound_atom_name='C5',
        angle_atom_name='C4',
        torsion_atom_name='C3',
        distance=0.98,
        angle=109.47,
        torsion_angle_add=240,
        refine=True
    )

Of course a lot of other cases can also be dealt with with the TorsionPositionConstraint
such as two sp2 hydrogen atoms, hydroxy groups or many more.

One sp2 hydrogen atom
*********************

For a single hydrogen atom located at an sp2 carbon, we can use the 
TrigonalPositionConstraint. Again the direction vectors of the two connecting atoms
to the atom bound directly to the hydrogen atoms are added up to generate the 
direction to the hydrogen atom.

.. code-block:: python

    constraint_dict['H5']['xyz'] = TrigonalPositionConstraint(
        bound_atom_name='C5',
        plane_atom1_name='C4',
        plane_atom2_name='C6',
        distance=0.95,
    )

The implementation is certainly slower than usual iam refinement routines, but
are still available if needed. However, usually we *aim* to determine hydrogen
positions by Hirshfeld Atom Refinement. For this determination, position
constraints are not applicable, of course.

For an example with hydrogen atoms placed by constraints look into the 
L-Alanin_iam_HConstraints folder within the examples.