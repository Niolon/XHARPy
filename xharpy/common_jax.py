import contextlib
import warnings
import logging

#logging.basicConfig(filename='xharpy.log', encoding='utf-8', level=logging.ERROR)

try:
    with contextlib.redirect_stdout(None):
        # supress only cpu message
        import jax
        import jax.numpy as jnp

except ImportError:
    import numpy as jnp
    import unittest.mock as jax
    warnings.warn('Jax was not found. Refinement will not be available')

__all__ = ['jax', 'jnp']
