from typing import Dict, Any, Union

XHARPY_VERSION = '0.2.0'

def get_value_or_default(
    parameter_name: str,
    refinement_dict: Dict[str, Any]
) -> Any:
    """Central function for storing the default values of the refinement_dict
    values. Always getting the default from here, means assumptions about
    defaults will be the same throughout the library.

    Parameters
    ----------
    parameter_name : str
        Parameter that will be looked up
    refinement_dict : Dict[str, Any]
        Dict with parameters for the refinement.

    Returns
    -------
    return_value: Any
        If the value is present in the refinement_dict, the value will be 
        returned from there. Otherwise the value from the defaults dictionary 
        will be returned
    """
    defaults = {
        'f0j_source': 'gpaw',
        'reload_step': 1,
        'core': 'constant',
        'extinction': 'none',
        'max_dist_recalc': 1e-6,
        'max_iter': 100,
        'min_iter': 10,
        'restraints': [],
        'flack': False,
        'core_io': ('none', 'core.pic'),
        'cutoff': ('none', 'none', 0.0)
    }
    return refinement_dict.get(parameter_name, defaults[parameter_name])



def get_parameter_index(
    parameter_name: str, 
    refinement_dict: Dict[str, Any]
) -> Union[int, None]:
    """Return the index of the parameter for the given refinement dict

    Parameters
    ----------
    parameter_name : str
        Name of the parameter, which is to be looked up
    refinement_dict : Dict[str, Any]
        Dict with parameters for the refinement.

    Returns
    -------
    index: Union[int, None]
        Integer with the index of the parameter in the parameter array if the
        value is actually refined. None if value is not refined.
    """
    order = [
        ('overall scaling', True),
        ('flack', get_value_or_default('flack', refinement_dict)),
        ('core', get_value_or_default('core', refinement_dict) == 'scale'),
        ('extinction', get_value_or_default('extinction', refinement_dict) != 'none')
    ]
    index = [name for name, _ in order].index(parameter_name)
    if not order[index][1]:
        return None
    return sum([int(val) for _, val in order][:index])