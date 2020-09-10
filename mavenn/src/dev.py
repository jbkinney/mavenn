# Standard imports
import numpy as np
import pandas as pd

# MAVE-NN imports
from mavenn.src.validate import validate_seqs, validate_alphabet, \
    validate_1d_array
from mavenn.src.error_handling import check, handle_errors
from mavenn.src.validate import alphabet_dict

@handle_errors
def x_to_alphabet(x, return_name=False):
    """
    Return the alphabet from which a set of sequences are drawn.

    parameters
    ----------
    x: (array of str)
        Array of sequences (equal-length strings).

    return_name: (bool)
        Whether to return the name of the alphabet, as opposed to an np.array
        of characters.

    returns
    -------
    alphabet: (np.array or str)
        If return_name=False, is an np.array of the characters in the identified
        alphabet. If no match is found with a pre-defined alphabet, returns
        a sorted array of unique characters in x. If return_name=True,
        is a str representing the name of the alphabet. If no pre-defined
        alphabet match exists, will be set to 'unknown'.
    """

    # Validate set of sequences with unknown alphabet
    x = validate_seqs(x, alphabet=None)

    # Get set of unique characters as the empirical alphabet
    empirical_alphabet = np.array(list(set(''.join(x))))
    empirical_alphabet.sort()

    # Find smallest matching alphabet if it exists
    alphabet = empirical_alphabet
    loss = np.inf
    name = 'unknown'
    for n, a in alphabet_dict.items():
        if set(a) >= set(empirical_alphabet) and len(a) < loss:
            alphabet = a
            loss = len(a)
            name = n

    # Return the alphabet
    if return_name:
        return name
    else:
        return alphabet

@handle_errors
def x_to_consensus(x, weights=None):
    """
    Identify the consensus sequence from a sequence alignment.

    parameters
    ----------
    x: (np.ndarray)
        List of sequences. Sequences must all be the same length.

    weights: (np.array)
        Weights for each sequence. E.g., count values, or numerical y values.
        If none, a value of 1 will be assumed for each sequence.

    returns
    -------
    consensus_seq: (str)
        Consensus sequence
    """

    # Validate sequences
    x = validate_seqs(seqs=x,
                      alphabet=None,
                      restrict_seqs_to_alphabet=False)

    # Check weights and set if not provided
    if weights is None:
        weights = np.ones(len(x))
    else:
        weights = validate_1d_array(weights)
        weights = weights.astype(float)
        check(len(weights) == len(x),
              f'len(weights)={len(weights)} does not match len(x)={len(x)}')

    # Split strings, creating a matrix of individual characters
    seqs = np.array([list(s) for s in x])

    # For each position, identify the most common character and store in list
    L = seqs.shape[1]
    poss = [l for l in range(L)]
    df = pd.DataFrame(data=seqs, columns=poss)
    df['weights'] = weights
    consensus_chars = []
    for l in poss:
        sub_cols = [l, 'weights']
        char_df = df[sub_cols].groupby(l).sum()
        char_df = char_df.sort_values('weights', ascending=False)
        consensus_chars.append(char_df.index[0])

    # Convert from list to seq
    consensus_seq = ''.join(consensus_chars)

    return consensus_seq


@handle_errors
def x_to_missing(x, alphabet=None):
    """
    Identify the missing characters at each position within a set
    of sequences.

    parameters
    ----------

    x: (np.ndarray)
        List of sequences. Sequences must all be the same length.

    alphabet: (string or list of characters)
        The alphabet from which characters in x are expected to be drawn.
        If None, the alphabet will be assumed to comprise the unique characters
        at all positions in all sequences in x.

    returns
    -------

    missing_dict: (dict)
        Keys denote positions, values are strings comprised of missing
        characters.
    """

    # If alphabet is not specified, determine automatically.
    if alphabet is None:
        alphabet = x_to_alphabet(x)

    # Validate alphabet
    alphabet = validate_alphabet(alphabet)

    # Validate sequences
    x = validate_seqs(seqs=x,
                      alphabet=alphabet,
                      restrict_seqs_to_alphabet=False)

    # Split strings, creating a matrix of individual characters
    seqs = np.array([list(s) for s in x])

    # For each position, identify missing characters and use to make mask_dict
    L = seqs.shape[1]
    missing_dict = {}
    for l in range(L):
        missing_chars = list(set(alphabet) - set(seqs[:, l]))
        missing_chars.sort()
        if len(missing_chars) > 0:
            missing_dict[l] = ''.join(missing_chars)

    return missing_dict
