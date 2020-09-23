# Standard imports
import numpy as np
import time

# MAVE-NN imports
from mavenn.src.validate import validate_seqs, validate_alphabet
from mavenn.src.error_handling import handle_errors, check

@handle_errors
def x_to_ohe(x,
             alphabet,
             check_seqs=True,
             check_alphabet=True,
             ravel_seqs=True):
    """
    parameters
    ----------
    x: (np.ndarray)
        (N,) array of input sequences, each of length L

    alphabet: (np.ndarray)
        (C,) array describing the alphabet sequences are drawn from.

    check_seqs: (bool)
        Whether to validate the sequences

    check_alphabet: (bool)
        Whether to validate the alphabet

    ravel_seqs: (bool)
        Whether to return an (N, L*C) array, as opposed to an (N, L, C) array.

    returns
    -------
    x_ohe: (np.ndarray)
        Array of one-hot encoded sequences, stored as np.int8 values.
    """

    # Validate alphabet as (C,) array
    if check_alphabet:
        alphabet = validate_alphabet(alphabet)

    # Validate sequences as (N,) array
    if check_seqs:
        x = validate_seqs(x, alphabet=alphabet)

    # Get dimensions
    L = len(x[0])
    N = len(x)
    C = len(alphabet)

    # Shape sequences as array of int8s
    x_arr = np.frombuffer(bytes(''.join(x), 'utf-8'),
                          np.int8, N * L).reshape([N, L])

    # Create alphabet as array of int8s
    alphabet_arr = np.frombuffer(bytes(''.join(alphabet), 'utf-8'),
                                 np.int8, C)

    # Compute (N,L,C) grid of one-hot encoded values
    x_nlc = (x_arr[:, :, np.newaxis] ==
             alphabet_arr[np.newaxis, np.newaxis, :]).astype(np.int8)

    # Ravel if requested
    if ravel_seqs:
        x_ohe = x_nlc.reshape([N, L * C])
    else:
        x_ohe = x_nlc

    return x_ohe






