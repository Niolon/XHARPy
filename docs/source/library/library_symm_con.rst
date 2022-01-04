Special Position Constraints
============================

The easiest way to generate special position constraints is by using the 
``lst2constraint_dict`` function. If there is no .lst available you can also
write the constraits by hand using the ``ConstrainedValues`` namedTuples.
If you want to see a finished example where constraints are used for 
Gram-Charlier parameters look at Urea\_gpaw in the examples folder.

So in a first step we need to import These

.. code-block:: python

    import ConstrainedValues

Let us say we have an atom O2 located on a two-fold axis with the special
position ``0.5 y 0.5``. This means the position is constrained. If we do not
have a constraint_dict we want to modify we would generate a new one.

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

Finally, the orderings of the individual parameters is as follows:

===== =======================================================================================================
name  order
===== =======================================================================================================
xyz   x, y, z

uij   U11, U22, U33, U23, U13, U12

cijk  C111, C222, C333, C112, C122, C113, C133, C223, C233, C123

dijkl D1111, D2222, D3333, D1112, D1222, D1113, D1333, D2223, D2333, D1122, D1133, D2233, D1123, D1223, D1233

occ   occ
===== =======================================================================================================

