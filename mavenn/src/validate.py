# Standard imports
import numpy as np
import pandas as pd

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


@handle_errors
def validate_1d_array(x):
    """
    Casts x as a 1d numpy array
    """

    # Get shape and cast as 1d
    x, shape = _get_shape_and_return_1d_array(x)

    return x


@handle_errors
def validate_nd_array(x):
    """
    Casts x as a numpy array of the original input shape.
    """

    # Get shape and cast as 1d
    x, shape = _get_shape_and_return_1d_array(x)

    # Return to original shape
    x = x.reshape(shape)

    return x

# TODO: 'protein*' will break current regular expressions. Those need to change.
@handle_errors
def validate_alphabet(alphabet):
    """
    Returns a validated alphabet. String inputs are interpreted
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
def validate_seqs(seqs, alphabet, restrict_seqs_to_alphabet=True):
    """
    Makes sure that seqs is an array of equal-length sequences
    drawn from the set of characters in alphabet. Returns
    a version of seqs cast as a numpy array of strings.
    """

    # Cast as np.array
    if isinstance(seqs, str):
        seqs = np.array([seqs])
    elif isinstance(seqs, (list, np.ndarray)):
        seqs = np.array(seqs).astype(str)
    elif isinstance(seqs, pd.Series):
        seqs = seqs.values.astype(str)
    else:
        check(False, f'type(seqs)={type(seqs)} is invalid.')

    # Make sure array is 1D
    check(len(seqs.shape) == 1, f'seqs should be 1D; seqs.shape={seqs.shape}')

    # Get length and make sure its >= 1
    N = len(seqs)
    check(N >= 1, f'N={N} must be >= 1')

    # Make sure all seqs are the same length
    lengths = np.unique([len(seq) for seq in seqs])
    check(len(lengths) == 1,
          f"Sequences should all be the same length"
          "; found multiple lengths={lengths}")

    # Make sure sequences only contain characters in alphabet
    if restrict_seqs_to_alphabet:
        seq_chars = set(''.join(seqs))
        alphabet_chars = set(alphabet)
        check(seq_chars <= alphabet_chars,
              f"seqs contain the following characters not in alphabet:"
              "{seq_chars-alphabet_chars}")

    return seqs


# @handle_errors
# def validate_input(df):
#     """
#     Checks to make sure that the input dataframe, df, contains
#     sequences and values that are valid for mavenn. Sequences
#     must all be of the same length and values have to be
#     integers or floats and not nan or string etc.
#
#     parameters
#     ----------
#
#     df: (dataframe)
#         A pandas dataframe containing two columns: (i) sequences and (ii)
#         values.
#
#     returns
#     -------
#     out_df: (dataframe)
#         A cleaned-up version of df (if possible).
#     """
#
#     # check that df is a valid dataframe
#     check(isinstance(df, pd.DataFrame),
#           'Input data needs to be a valid pandas dataframe, '
#           'input entered: %s' % type(df))
#
#     # create copy of df so we don't overwrite the user's data
#     out_df = df.copy()
#
#     # make sure the input df has only sequences and values
#     # and no additional columns
#     check(out_df.shape[1] == 2, 'Input dataframe must only have 2 columns, '
#                                 'sequences and values. Entered # columns %d' % len(out_df.columns))
#
#     # check that 'sequences' and 'values columns are part of the df columns
#     check('sequence' in out_df.columns, 'Column containing sequences must be named "sequence" ')
#     check('values' in out_df.columns, 'Column containing values must be named "values" ')
#
#     # TODO: check that sequence column is of type string and values column is float or int
#     # TODO: need to check that sequences are of the same length, but this could take a lot of time checking
#
#     # return cleaned-up out_df
#     out_df = out_df.dropna().copy()
#     return out_df