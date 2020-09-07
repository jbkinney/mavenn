# Standard imports
import numpy as np
import pandas as pd

# Imports from MAVE-NN
from mavenn.src.validate import validate_alphabet
from mavenn.src.error_handling import handle_errors, check


@handle_errors
def get_1pt_variants(wt_seq, alphabet, include_wt=True):
    """
    Returns a list of all single-point mutants of a given wilde-type sequence.

    parameters
    ----------
        wt_seq: (str)
            The wild-type sequence. Must comprise characters from alphabet.

        alphabet: (str, array-like)
            The alphabet (name or list of characters) to use for mutations.

        include_wt: (bool)
            Whether to include the wild-type sequence in the output.

    returns
    -------

        out_df: (pd.DataFrame)
            A dataframe listing each variant sequence along with its name,
            the position mutated, the wild-type character at that position,
            and the mutant character at that position. Characters ' ' and
            a position of -1 is used for the wild-type sequence if
            include_wt is True.
    """

    # Check that wt_seq is a string
    check(isinstance(wt_seq, str),
          f'wt_seq must be a string; is of type {type(wt_seq)}')
    L = len(wt_seq)

    # Check that include_wt is bool
    check(isinstance(include_wt, bool),
          f'type(include_wt)={type(include_wt)}; must be bool.')

    # Check length
    check(L >= 1,
          f'len(wt_seq)={L}; must be >= 1.')

    # Create a list of all single-point variants of wt sequence
    alphabet = validate_alphabet(alphabet)
    C = len(alphabet)

    # Make sure wt_seq comprises alphabet
    seq_set = set(list(wt_seq))
    alphabet_set = set(alphabet)
    check(seq_set <= alphabet_set,
          f"wt_seq={wt_seq} contains the invalid characters {seq_set-alphabet_set}")

    # Create a list of seqs with all single amino acid changes at all positions
    pos = np.arange(L).astype(int)
    cs = list(np.tile(np.reshape(alphabet, [1, C]), [L, 1]).ravel())
    ls = list(np.tile(np.reshape(pos, [L, 1]), [1, C]).ravel())
    cs_wt = [wt_seq[l] for c, l in zip(cs, ls)]
    seqs = [wt_seq[:l] + c + wt_seq[l + 1:] for c, l in zip(cs, ls)]
    names = [wt_seq[l] + str(l) + c for c, l in zip(cs, ls)]
    ix = [wt_seq[l] != c for c, l in zip(cs, ls)]

    # Include wt if requested
    if include_wt:
        names = ['WT'] + names
        cs_wt = [''] + cs_wt
        cs = [''] + cs
        ls = [-1] + ls
        seqs = [wt_seq] + seqs
        ix = [True] + ix

    # Report results in the form of a dataframe
    out_df = pd.DataFrame()
    out_df['name'] = names
    out_df['l'] = ls
    out_df['c_wt'] = cs_wt
    out_df['c_mut'] = cs
    out_df['seq'] = seqs

    # Remove sequences identical to wt
    out_df = out_df[ix]
    out_df.set_index('name', inplace=True)

    # Return seqs
    return out_df


@handle_errors
def get_2pt_variants(wt_seq, alphabet, include_wt=True):
    """
    Returns a list of all twp-point mutants of a given wilde-type sequence.

    parameters
    ----------
        wt_seq: (str)
            The wild-type sequence. Must comprise characters from alphabet.

        alphabet: (str, array-like)
            The alphabet (name or list of characters) to use for mutations.

        include_wt: (bool)
            Whether to include the wild-type sequence in the output.

    returns
    -------

        out_df: (pd.DataFrame)
            A dataframe listing each variant sequence along with its name,
            the positions mutated, the wild-type characters at each position
            pair, and the mutant characters at those positions.
            Empty strings and positions of -1 are used for the
            wild-type sequence if include_wt=True.
    """

    # Check that wt_seq is a string
    check(isinstance(wt_seq, str),
          f'wt_seq must be a string; is of type {type(wt_seq)}')
    L = len(wt_seq)

    # Check that include_wt is bool
    check(isinstance(include_wt, bool),
          f'type(include_wt)={type(include_wt)}; must be bool.')

    # Check length
    check(L >= 1,
          f'len(wt_seq)={L}; must be >= 1.')

    # Create a list of all single-point variants of wt sequence
    alphabet = validate_alphabet(alphabet)
    C = len(alphabet)

    # Make sure wt_seq comprises alphabet
    seq_set = set(list(wt_seq))
    alphabet_set = set(alphabet)
    check(seq_set <= alphabet_set,
          f"wt_seq={wt_seq} contains the invalid characters {seq_set-alphabet_set}")

    # Create a list of seqs with all single amino acid changes at all positions
    pos = np.arange(L).astype(int)
    l1s = np.tile(np.reshape(pos, [L, 1, 1, 1]), [1, C, L, C]).ravel()
    c1s = np.tile(np.reshape(alphabet, [1, C, 1, 1]), [L, 1, L, C]).ravel()
    l2s = np.tile(np.reshape(pos, [1, 1, L, 1]), [L, C, 1, C]).ravel()
    c2s = np.tile(np.reshape(alphabet, [1, 1, 1, C]), [L, C, L, 1]).ravel()

    # Remove indices for unallowed combinations of ls
    ix = l2s > l1s
    l1s = list(l1s[ix].astype(int))
    c1s = list(c1s[ix])
    l2s = list(l2s[ix].astype(int))
    c2s = list(c2s[ix])

    # Create all pairwise variants
    seqs = [wt_seq[:l1] + c1 + wt_seq[l1 + 1:l2] + c2 + wt_seq[l2:]
            for (l1, c1, l2, c2) in zip(l1s, c1s, l2s, c2s)]

    names = [wt_seq[l1] + str(l1) + c1 + ',' + wt_seq[l2] + str(l2) + c2
             for l1, c1, l2, c2 in zip(l1s, c1s, l2s, c2s)]

    ix = [(wt_seq[l1] != c1) & (wt_seq[l2] != c2)
          for l1, c1, l2, c2 in zip(l1s, c1s, l2s, c2s)]

    c1s_wt = [wt_seq[l1]
              for l1, c1, l2, c2 in zip(l1s, c1s, l2s, c2s)]

    c2s_wt = [wt_seq[l2]
              for l1, c1, l2, c2 in zip(l1s, c1s, l2s, c2s)]

    # Include wt if requested
    if include_wt:
        names = ['WT'] + names
        c1s_wt = [''] + c1s_wt
        c1s = [''] + c1s
        l1s = [-1] + l1s
        c2s_wt = [''] + c2s_wt
        c2s = [''] + c2s
        l2s = [-1] + l2s
        seqs = [wt_seq] + seqs
        ix = [True] + ix

    # Report results in the form of a dataframe
    out_df = pd.DataFrame()
    out_df['name'] = names
    out_df['l1'] = l1s
    out_df['c1_wt'] = c1s_wt
    out_df['c1_mut'] = c1s
    out_df['l2'] = l2s
    out_df['c2_wt'] = c2s_wt
    out_df['c2_mut'] = c2s
    out_df['seq'] = seqs

    # Remove sequences identical to wt
    out_df = out_df[ix]
    out_df.set_index('name', inplace=True)

    # Return seqs
    return out_df