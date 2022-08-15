import contextlib
import warnings
import logging

logging.basicConfig(filename='xharpy.log', encoding='utf-8', level=logging.ERROR)

try:
    with contextlib.redirect_stdout(None):
        # supress only cpu message
        import jax.numpy as jnp
        import jax
    from jax.config import config
    try:
        config.update('jax_enable_x64', True)
    except:
        warnings.warn('Could not activate 64 bit mode of jax. Might run in 32 bit instead.')

except:
    import numpy as jnp
    import unittest.mock as jax
    warnings.warn('Jax was not found. Refinement will not be available')
