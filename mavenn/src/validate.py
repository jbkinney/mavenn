"""validate.py: Utilities for validating input."""

# Standard imports
import numpy as np
import pandas as pd
import pdb

# MAVE-NN imports
from mavenn.src.reshape import _get_shape_and_return_1d_array
from mavenn.src.error_handling import check, handle_errors

# Define built-in alphabets to use with MAVE-NN
alphabet_dict = {
    'dna': np.array(['A', 'C', 'G', 'T']),
    'rna': np.array(['A', 'C', 'G', 'U']),
    'protein': np.array(['A', 'C', 'D', 'E', 'F',
                         'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R',
                         'S', 'T', 'V', 'W', 'Y']),
    'protein*': np.array(['A', 'C', 'D', 'E', 'F',
                          'G', 'H', 'I', 'K', 'L',
                          'M', 'N', 'P', 'Q', 'R',
                          'S', 'T', 'V', 'W', 'Y', '*'])
}

# Translate from amino acid abbreviations to single letter symbols.
abreviation_dict = {
    'Ala': 'A',
    'Arg': 'R',
    'Asn': 'N',
    'Asp': 'D',
    'Cys': 'C',
    'Glu': 'E',
    'Gln': 'Q',
    'Gly': 'G',
    'His': 'H',
    'Ile': 'I',
    'Leu': 'L',
    'Lys': 'K',
    'Met': 'M',
    'Phe': 'F',
    'Pro': 'P',
    'Ser': 'S',
    'Thr': 'T',
    'Trp': 'W',
    'Tyr': 'Y',
    'Val': 'V'
}


@handle_errors
def validate_1d_array(x):
    """Cast x as a 1d numpy array."""
    # Get shape and cast as 1d
    x, shape = _get_shape_and_return_1d_array(x)

    return x


@handle_errors
def validate_nd_array(x):
    """Casts x as a numpy array of the original input shape."""
    # Get shape and cast as 1d
    x, shape = _get_shape_and_return_1d_array(x)

    # Return to original shape
    x = x.reshape(shape)

    return x


# TODO: 'protein*' will break current regular expressions. Those need to change.
@handle_errors
def validate_alphabet(alphabet):
    """
    Return a validated alphabet.

    String inputs are interpreted
    as the name of one of four alphabets:
        ['dna','rna','protein','protein*'].
    Otherwise alphabet must be one of
        [set, list, np.ndarray, pd.Series],
    containing only unique characters.
    """
    valid_types = (str, list, set, np.ndarray, pd.Series)
    check(isinstance(alphabet, valid_types),
          f'type(alphabet)={type(alphabet)} is invalid. '
          f'Must be one of {valid_types}.')

    # If alphabet is a string, replace with array from alphabet_dict
    if isinstance(alphabet, str):
        check(alphabet in alphabet_dict,
              f'Unknown alphabet={alphabet}. Must be one of [{alphabet_dict.keys()}].')
        alphabet = alphabet_dict[alphabet]

    # If alphabet is a set, cast as np.ndarray
    elif isinstance(alphabet, set):
        alphabet = np.array(list(alphabet))

    # If alphabet is a list, cast an np.ndarray
    elif isinstance(alphabet, list):
        alphabet = np.array(alphabet)

    # If alphabet is a pd.Series, get values
    elif isinstance(alphabet, pd.Series):
        alphabet = alphabet.values

    # Make sure alphabet is 1D
    check(len(alphabet.shape) == 1,
          f'Alphabet must be 1D. alphabet.shape={alphabet.shape}')

    # Make sure the entries of alphabet are unique
    check(len(alphabet) == len(set(alphabet)),
          f'Entries of alphabet are not unique.')

    # Make sure alphabet is not empty
    check(len(alphabet) > 0,
          f'len(alphabet)={len(alphabet)}; must be >= 1.')

    # Make sure all alphabet entries are strings
    check(all([isinstance(c, str) for c in alphabet]),
          'Alphabet contains non-string characters.')

    # Make sure all alphabet entries are single-character
    check(all([len(c) == 1 for c in alphabet]),
          'Alphabet contains non-string characters.')

    # Sort alphabet
    alphabet.sort()

    return alphabet


@handle_errors
def validate_seqs(x,
                  alphabet=None,
                  restrict_seqs_to_alphabet=True):
    """
    Validate sequences for use in MAVE-NN.

    Makes sure that seqs is an array of equal-length sequences
    drawn from the set of characters in alphabet. Returns
    a version of seqs cast as a numpy array of strings. Note that
    alphabet must be set when setting restrict_seqs_to_alphabet=True.

    Parameters
    ----------
    x: (array-like)
        Array of equal-length sequences.

    alphabet: (str, array-like)
        Alphabet from which strings are drawn.

    restrict_seqs_to_alphabet: (bool)
        Whether to restrict sequences to the specified alphabet.

    Returns
    -------
    x: (np.array)
        Nrray of validated sequences
    """
    # Cast as np.array
    if isinstance(x, str):
        x = np.array([x])
    elif isinstance(x, (list, np.ndarray)):
        x = np.array(x).astype(str)
    elif isinstance(x, pd.Series):
        x = x.values.astype(str)
    else:
        check(False, f'type(x)={type(x)} is invalid.')

    # Make sure array is 1D
    check(len(x.shape) == 1, f'x should be 1D; x.shape={x.shape}')

    # Get N and make sure its >= 1
    N = len(x)
    check(N >= 1, f'N={N} must be >= 1')

    # Make sure all x are the same length
    lengths = np.unique([len(seq) for seq in x])
    check(len(lengths) == 1,
          f"Sequences should all be the same length"
          "; found multiple lengths={lengths}")

    # If user requests to restrict sequences to a given alphabet
    if restrict_seqs_to_alphabet:

        # Check that alphabet is specified
        check(alphabet is not None,
              "alphabet must be specified when restrict_seqs_to_alphabet=True.")

        # Validate alphabet
        alphabet = validate_alphabet(alphabet)

        # Make sure all sequences are in alphabet
        seq_chars = set(''.join(x))
        alphabet_chars = set(alphabet)
        check(seq_chars <= alphabet_chars,
              f"x contain the following characters not in alphabet:"
              "{seq_chars-alphabet_chars}")

    return x
