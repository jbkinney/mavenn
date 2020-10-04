"""reshape.py: Utilities for reshaping input into np.arrays."""

# Standard imports
import numpy as np
from collections.abc import Iterable

# Imports from MAVE-NN
from mavenn.src.error_handling import handle_errors, check

@handle_errors
def _broadcast_arrays(x, y):
    """Broadcast arrays."""
    # Cast inputs as numpy arrays
    # with nonzero dimension
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    # Get shapes
    xshape = list(x.shape)
    yshape = list(y.shape)

    # Get singltons that mimic shapes
    xones = [1] * x.ndim
    yones = [1] * y.ndim

    # Broadcast
    x = np.tile(np.reshape(x, xshape + yones), xones + yshape)
    y = np.tile(np.reshape(y, xones + yshape), xshape + yones)

    # Return broadcast arrays
    return x, y


@handle_errors
def _get_shape_and_return_1d_array(x):
    """Get shape and return 1D array."""
    if not isinstance(x, Iterable):
        shape = []
    else:
        x = np.array(x)
        shape = list(x.shape)
    x = np.atleast_1d(x).ravel()
    return x, shape


@handle_errors
def _shape_for_output(x, shape=None):
    """Shape array for output."""
    if shape is not None:
        x = np.array(x)
        x = np.reshape(x, shape)
    else:
        x = np.squeeze(x)
    if x.ndim == 0:
        x = x.tolist()
    return x
