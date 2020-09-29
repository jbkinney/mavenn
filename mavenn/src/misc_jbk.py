# Standard imports
import numpy as np
import pandas as pd
import re
import tempfile
import os
import tensorflow as tf
import pdb
import time
import logomaker

# Scipy imports
from scipy.stats import expon

# Tensorflow imports
import tensorflow.keras.backend as K
#from tensorflow.keras.constraints import non_neg
from tensorflow.keras.initializers import Constant
#from tensorflow.math import tanh, sigmoid

# MAVE-NN imports
from mavenn.src.validate import validate_seqs, \
                                validate_alphabet, \
                                validate_1d_array
from mavenn.src.error_handling import check, handle_errors
from mavenn.src.validate import alphabet_dict
from mavenn.src.reshape import _get_shape_and_return_1d_array
from mavenn.src.sequence_features import x_to_ohe


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
def x_to_features(x,
                  alphabet,
                  model_type,
                  batch_size=1000,
                  max_mem=1E9,
                  restrict_seqs_to_alphabet=True):
    """
    Encodes sequences as sequence features.

    parameters
    ----------
    x: (np.ndarray)
        1D numpy array of sequences. Each element must be a string of
        identical length L.

    alphabet: (str)
        Name of alphabet to use.

    model_type: (str)
        Type of model. Must be one of "additive", "neighbor", "pairwise".

    batch_size: (int)
        Number of sequences to encode simultaneously.

    max_mem: (int, float)
        Maximum memory to allocate for output. If output exceeds this memory,
        an error will be raised.

    restrict_seqs_to_alphabet: (bool)
        Whether to throw an error if any sequences have a character not in
        the alphabet. If False, unrecognized characters will be encoded as 0.

    returns
    -------

    x_bin: (np.array of int8)
        2D numpy array of binary sequence features. Shape is (N,K), where
        N is the number of sequences and K is the number of sequence features.

    names: (list)
        A list of strings, giving the names of the features encoded in x_bin.
    """

    # Cast x as a 1D array. Get shape in order to reshape output
    x, in_shape = _get_shape_and_return_1d_array(x)
    N = len(x)

    # Check model_type
    valid_model_types = ['additive', 'neighbor', 'pairwise']
    check(model_type in valid_model_types,
          f'model_type={repr(model_type)}; must be one of {valid_model_types}.')

    # Check alphabet
    alphabet = validate_alphabet(alphabet)

    # Get number of characters
    C = len(alphabet)

    # Check batch_size
    check(isinstance(batch_size, int),
          f'type(batch_size)={type(batch_size)}; must be int.')
    check(batch_size > 0,
          f'batch_size={batch_size}; must be > 0.')

    # Check max_mem
    check(isinstance(max_mem, (int, float)),
          f'type(max_mem)={type(max_mem)}; must be int or float.')

    # Validate sequences
    x = validate_seqs(x,
                      alphabet=alphabet,
                      restrict_seqs_to_alphabet=restrict_seqs_to_alphabet)

    # Get sequence length
    L = len(x[0])

    # Get the number of features
    if model_type == 'additive':
        K = 1 + C*L
    elif model_type == 'neighbor':
        K = 1 + C*L + C*C*(L-1)
    elif model_type == 'pairwise':
        K = 1 + C*L + C*C*L*(L-1)/2
    else:
        assert False, 'This should not happen.'
    K = int(K)

    # Make sure that max_mem is not exceeded

    check(N * K <= max_mem,
          f'Aborting; would require storing {N * K:.2g} values in'
          f'representing x; must be <= max_mem={max_mem:.2g}.')

    # Allocate memory
    x_bin = np.zeros([N, K], dtype=np.int8)

    # Fill in x_bin by batch
    start = 0
    stop = min(start+batch_size, N)
    while start < N:

        # Take batch of sequences
        x_batch = x[start:stop]

        # Encode batch
        if model_type == 'additive':
            x_batch_bin, names = \
                additive_model_features(x_batch,
                                        alphabet,
                                        restrict_seqs_to_alphabet)
        elif model_type == 'neighbor':
            x_batch_bin, names = \
                neighbor_model_features(x_batch,
                                        alphabet,
                                        restrict_seqs_to_alphabet)
        elif model_type == 'pairwise':
            x_batch_bin, names = \
                pairwise_model_features(x_batch,
                                        alphabet,
                                        restrict_seqs_to_alphabet)
        else:
            assert False, 'This should not happen.'

        # Make sure sizes are correct
        assert x_batch_bin.shape[1] == K, 'This should not happen.'

        # Store batch
        x_bin[start:stop] = x_batch_bin

        # Reset batch bounds
        start = stop
        stop = min(start+batch_size, N)

    # Reshape encoded sequences and return
    out_shape = list(in_shape) + [K]
    x_bin = x_bin.reshape(out_shape)

    return x_bin, names


@handle_errors
def set_seed(seed):
    """
    Set seed in order to make results reproducible.

    parameters
    -------
    seed: (int)
        The value provided is used in both np.random.seed()
        and tf.random.set_seed().

    returns
    -------
    None

    """

    # Check seed
    check(isinstance(seed, int),
          f'type(seed)={type(seed)}; must be int')

    # Set seed
    np.random.seed(seed)
    tf.random.set_seed(seed)


@handle_errors
def mavedb_to_dataset(df,
                      hgvs_col,
                      y_col,
                      other_cols=None,
                      training_frac=.8,
                      seed=0,
                      dropna_y=True):
    """
    Converts a variants dataset from MaveDB to a MAVE-NN compatible datset.

    parameters
    ----------
    df: (pd.DataFrame)
        A pandas dataframe containing by-variant data from MaveDB.

    hgvs_col: (str)
        Name of column containing variant information.

    y_col: (str)
        Name of column in df to use as y-values.

    other_cols: (None, list)
        List of names of other columns to include in dataset.
        If None, all columns other than y_col and hgvs_col will be used.

    training_frac: (float in [0,1])
        Fraction of data to mark for training set.

    seed: (int)
        Argument to pass to np.random.seed() before setting
        training set flags.

    dropna_y: (bool)
        Remove rows that contain NA values in y_col.
    """

    ### Perform checks

    # Check hgvs_col
    check(isinstance(hgvs_col, str),
          f'type(hgvs_col)={type(hgvs_col)}; must be str')
    check(hgvs_col in df.columns,
          f'hgvs_col={hgvs_col} is not in df.columns={df.columns}')

    # Check y_col
    check(isinstance(y_col, str),
          f'type(y_col)={type(y_col)}; must be str')
    check(y_col in df.columns,
          f'y_col={y_col} is not in df.columns={df.columns}')

    # Of other_cols is None, set equal to df columns other than ones above
    if other_cols is None:
        other_cols = [col for col in df.columns if col not in [hgvs_col, y_col]]

    # If other_cols is list, make sure its a subset of df.columns
    elif isinstance(other_cols, list):
        check(set(other_cols) <= set(df.columns),
              f'y_col={other_cols} is not in df.columns={df.columns}')

    # Otherwise, throw error
    else:
        check(False,
              f'type(other_cols)={type(other_cols)} is not None or list.')

    # Check training_frac
    check(isinstance(training_frac, float),
          f'type(training_frac)={type(training_frac)} is not float.')
    check((training_frac <= 1) and (training_frac >= 0),
          f'training_frac={training_frac} is not in [0,1]')

    # Check seed
    check(isinstance(seed, int),
          f'type(seed)={type(seed)}; must be int')

    # Check dropna_y
    check(isinstance(dropna_y, bool),
          f'type(dropna_y)={type(dropna_y)}')

    ### Compute y_df

    # Get number of variants
    N = len(df)

    # Create y_df
    y_df = pd.DataFrame()

    # Set training set and testing set with specified split
    np.random.seed(seed)
    training_flag = (np.random.rand(N) < training_frac)
    y_df['training_set'] = training_flag

    # Parse hgvs notation
    matches_list = [re.findall('([A-Za-z\*]+)([0-9]+)([A-Za-z\*]+)', s)
                    for s in df[hgvs_col]]

    # Add hamming_dist col to y_df
    y_df['hamming_dist'] = [len(m) for m in matches_list]

    # Set other columns
    y_df[other_cols] = df[other_cols].copy()

    # Set y
    y_df['y'] = df[y_col].copy()

    ### Parse hgvs_col values into mut_df

    # Parse strings in hgvs_col column.
    # There should be a better way than writing to a temp file
    fd, fname = tempfile.mkstemp(text=True)
    f = open(fname, 'w')
    f.write('id,l,c,c_wt\n')
    for i, matches in enumerate(matches_list):
        for c_wt, l, c in matches:
            f.write(f'{i},{int(l)-1},{c},{c_wt}\n')
    f.close()
    os.close(fd)
    mut_df = pd.read_csv(fname)
    os.remove(fname)

    # If any characters c have length 3, assume it's long protein
    # and do translation
    cs = list(mut_df['c']) + list(mut_df['c_wt'])
    if max([len(c) for c in cs]) == 3:


        # Remove all unrecognized 'c'
        ix = mut_df['c'].isin(abreviation_dict.keys())
        ix_wt = mut_df['c_wt'].isin(abreviation_dict.keys())
        mut_df = mut_df[ix & ix_wt]

        # Map long-form aa to short-form aa
        mut_df['c'] = mut_df['c'].map(abreviation_dict).astype(str)
        mut_df['c_wt'] = mut_df['c_wt'].map(abreviation_dict).astype(str)

        # Set alphabet
        alphabet = 'protein'

    # Otherwise, just capitalize, and remove any rows not in the alphabet
    else:

        # Capitalize
        mut_df['c'] = [c.upper() for c in mut_df['c']]
        mut_df['c_wt'] = [c.upper() for c in mut_df['c_wt']]
        cs = list(mut_df['c']) + list(mut_df['c_wt'])
        alphabet = x_to_alphabet(cs, return_name=True)

    ### Compute wt_seq from mut_df

    # Get unique list of lengths
    ls = mut_df['l'].unique()
    ls.sort()

    # Subtract minimum value; to take care of strange indexing
    ls -= ls.min()

    # Get sequence length
    L = ls.max() + 1

    # Fill in array; use '?' as missing character
    wt_seq_arr = np.array(['.'] * L)
    for l in ls:
        ix = mut_df['l'] == l
        c_wts = mut_df.loc[ix, 'c_wt'].unique()
        assert len(c_wts == 1)
        wt_seq_arr[l] = c_wts[0]
    wt_seq = ''.join(wt_seq_arr)

    ### Create data_df

    # Create dataset
    data_df = mutations_to_dataset(wt_seq=wt_seq, mut_df=mut_df, y_df=y_df)

    # Remove rows with NaN y-values if requested
    if dropna_y:
        ix_na = data_df['y'].isna()
        data_df = data_df[~ix_na]

    # Reindex
    data_df.reset_index(inplace=True, drop=True)
    data_df.index.name = 'id'

    ### Create info_dict

    # Add other useful information to info_dict
    info_dict = {}
    info_dict['wt_seq'] = wt_seq
    info_dict['alphabet'] = alphabet

    return data_df, info_dict


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
    name = "unknown"
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
def p_lc_to_x(N, p_lc, alphabet):
    """
    Generate an array of N sequences given a probability_df

    parameters
    ----------
    N: (int > 0)
        Number of sequences to generate

    p_lc: (np.array)
        An (L,C) array  listing the probability of each base (columns) for each
        position (rows).

    alphabet: (np.array)
        The alphabet, length C, from which sequences will be generated
    """

    # Validate N
    check(isinstance(N, int),
          f'type(N)={type(N)}; must be int')
    check(N > 0,
          f'N={N}; must be > 0')

    # Validate p_lc
    check(isinstance(p_lc, np.ndarray),
          f'type(p_lc)={type(p_lc)}; must be np.ndarray.')
    check(p_lc.ndim == 2,
          f'p_lc.ndim={p_lc.ndim}; must be 2.')
    check(np.all(p_lc.ravel() >= 0),
          f'some elements of p_lc are negative.')
    check(np.all(p_lc.ravel() <= 1),
          f'some elements of p_lc are greater than 1.')
    L, C = p_lc.shape

    # Validate alphabet
    alphabet = validate_alphabet(alphabet)
    check(len(alphabet) == C,
          f'len(alphabet)={len(alphabet)} does not match p_lc.shape[1]={C}')

    # Create function to fill in a set of characters given p_l
    def fill_x_arr_col(p_l):
        return np.random.choice(a=alphabet,
                                size=N,
                                replace=True,
                                p=p_l)

    # Create seqs
    x_arr = np.apply_along_axis(fill_x_arr_col, axis=1, arr=p_lc).T
    x = np.apply_along_axis(''.join, axis=1, arr=x_arr)

    return x

@handle_errors
def x_to_stats(x, alphabet, weights=None, verbose=False):
    """
    Identify the consensus sequence from a sequence alignment.

    parameters
    ----------
    x: (np.ndarray)
        List of sequences. Sequences must all be the same length.

    alphabet: (np.ndarray)
        Alphabet from which sequences are drawn.

    weights: (None, np.ndarray)
        Weights for each sequence. E.g., count values, or numerical y values.
        If None, a value of 1 will be assumed for each sequence.

    verbose: (bool)
        Whether to print computation time.

    returns
    -------
    consensus_seq: (str)
        Consensus sequence.
    """

    # Start timer
    start_time = time.time()

    # Validate alphabet
    alphabet = validate_alphabet(alphabet)

    # Validate sequences
    x = validate_seqs(seqs=x,
                      alphabet=alphabet,
                      restrict_seqs_to_alphabet=True)

    # Check weights and set if not provided
    if weights is None:
        weights = np.ones(len(x))
    else:
        weights = validate_1d_array(weights)
        weights = weights.astype(float)
        check(len(weights) == len(x),
              f"len(weights)={len(weights)} does not match len(x)={len(x)}")

    # Do one-hot encoding of sequences
    t = time.time()
    x_nlc = x_to_ohe(x,
                     alphabet,
                     check_seqs=False,
                     check_alphabet=False,
                     ravel_seqs=False)
    #print(f'Time for x_to_ohe: {time.time()-t:.3f} sec.')
    N, L, C = x_nlc.shape

    # Dictionary to hold results
    stats = {}

    # Compute x_ohe
    stats['x_ohe'] = x_nlc.reshape([N, L*C]).astype(np.int8)

    # Multiply by weights
    x_nlc = x_nlc.astype(float) * weights[:, np.newaxis, np.newaxis]

    # Compute lc encoding of consensus sequence
    x_sum_lc = x_nlc.sum(axis=0)
    x_sum_lc = x_sum_lc.reshape([L, C])
    x_support_lc = (x_sum_lc != 0)

    # Set number of sequences
    stats['N'] = N

    # Set sequence length
    stats['L'] = L

    # Set number of characters
    stats['C'] = C

    # Set alphabet
    stats['alphabet'] = alphabet

    # Compute probability matrix
    p_lc = x_sum_lc / x_sum_lc.sum(axis=1)[:, np.newaxis]
    stats['probability_df'] = pd.DataFrame(index=range(L),
                                           columns=alphabet,
                                           data=p_lc)

    # Compute sparsity factor
    stats['sparsity_factor'] = (x_nlc != 0).sum().sum() / x_nlc.size

    # Compute the consensus sequence and corresponding matrix.
    # Adding noise prevents ties
    x_sum_lc += 1E-1 * np.random.rand(*x_sum_lc.shape)
    stats['consensus_seq'] = \
        ''.join([alphabet[np.argmax(x_sum_lc[l, :])] for l in range(L)])

    # Compute mask dict
    missing_dict = {}
    for l in range(L):
        if any(~x_support_lc[l, :]):
            missing_dict[l] = ''.join(alphabet[~x_support_lc[l, :]])
    stats['missing_char_dict'] = missing_dict

    # Provide feedback if requested
    duration_time = time.time() - start_time
    if verbose:
        print(f'Stats computation time: {duration_time:.5f} sec.')

    return stats


@handle_errors
def mutations_to_dataset(wt_seq,
                         mut_df,
                         y_df=None,
                         l_col="l",
                         c_col="c",
                         id_col="id",
                         y_id_col=None,
                         y_keep_cols=None):
    """
    Compute an array of sequences (x) from a wild-type sequence (wt_seq) and a
    list of mutations (mut_df). mut_df is a dataframe that should have three
    columns:
     - l_col lists the positions of 1pt mutations,
     - c_col lists the mutant characters at those positions,
     - id_col lists numbers from 0 to N-1 that identifying each of the N variant
       sequences in which the mutations occur.
    Note that this last column, id_col, allows the listing of multiple 1pt
    mutations for each variant. This x is then combined with the values in y_df
    to create a MAVE-NN compatible dataset data_df, the indices of which are
    the variant ids.

    parameters
    ----------
    wt_seq: (str)
        The wild-type sequence

    mut_df: (pd.DataFrame)
        Dataframe containing mutation information.

    y_df: (pd.DataFrame)
        Dataframe containing measurement information, if any.

    l_col: (str)
        Name of mut_df colum listing positions of point mutaitons. Column
        values should be integers ranging from 0 to L-1.

    c_col: (str)
        Name of mut_df colum listing mutant characters at the listed positions.

    id_col: (str)
        Name of mut_df column listing the ID numbers for sequence variants.
        Column values should range from 0 to N-1 where N is the number of
        unique variants.

    y_id_col: (str)
        Name of column of y_df containing variant IDs. If None, will take
        values from y_df.index.

    y_keep_cols: (list-like)
        Columns in y_df to include in output. If None, all columns in y_df will
        be kept.

    returns
    -------
    data_df: (pd.DataFrame)
        Dataframe that list the full sequences of all variants (in column 'x'),
        as well as correpsonding measurements from y_df. Variant id numbers
        are used as index values.

    """

    # Check wt_seq is a string
    check(isinstance(wt_seq, str),
          f"type(wt_seq)={type(wt_seq)}; wt_seq must be of type str.")

    # Get length of wt sequence
    L = len(wt_seq)

    # Check mut_df is a dataframe
    check(isinstance(mut_df, pd.DataFrame),
          f"type(mut_df)=={type(mut_df)}; mut_df must be of type pd.DataFrame")

    # Check that all column names are in mut_df
    check(l_col in mut_df.columns,
          f"l_col={repr(l_col)} is not in mut_df.columns={mut_df.columns}")
    check(c_col in mut_df.columns,
          f"c_col={repr(c_col)} is not in mut_df.columns={mut_df.columns}")
    check(id_col in mut_df.columns,
          f"id_col={repr(id_col)} is not in mut_df.columns={mut_df.columns}")

    # Extract values and cast as relevant types
    ids = mut_df[id_col].values.astype(int)
    ls = mut_df[l_col].values.astype(int)
    cs = mut_df[c_col].values.astype(str)

    # Check that all characters are of length 1
    unique_characters = list(set(cs))
    check(all([len(c) == 1 for c in unique_characters]),
          f"Not all mutant characters are length 1, as required; "
          f"set(mut_df[{repr(c_col)}])={set(unique_characters)}")

    # Check that all positions are nonnegative
    check(min(ls) >= 0,
          f"Not all positions are nonnegative, as required; "
          f"min(mut_df[{repr(l_col)}])={min(ls)}")

    # Check that IDs are all nonnegative
    check(min(ids) >= 0,
          f'Not all ids are nonnegative, as required; '
          f'min(mut_df[{repr(id_col)}])={min(ids)}')

    # Get number of sequences
    N = max(ids) + 1

    # If user provides y_df
    if y_df is not None:

        # Make sure y_df is a dataframe
        check(isinstance(y_df, pd.DataFrame),
              f'type(y_df)={type(y_df)}; must be pd.DataFrame.')

        # If y_id_col is set, make this the index of y_df
        if y_id_col is not None:
            # Make sure y_id_col is in y.columns
            check(y_id_col in y.columns,
                  f'y_id_col={repr(y_id_col)} is not in y.columns={y.columns}')

            # Set as index
            y_df.set_index(y_id_col, drop=True)

        # If y_keep_cols is set
        if y_keep_cols is not None:
            # Make sure y_keep_cols is interable
            check(isinstance(y_keep_cols, Iterable),
                  f'type(y_keep_cols)={type(y_keep_cols)};'
                  f'must be an Iterable, such as a list.')

            # Make sure y_keep_cols contains columns in y_df
            check(all([c in y_df.columns for c in y_keep_cols]),
                  f'y_keep_cols={y_keep_cols} contains entries not '
                  f'in y_df.columns={y_df.columns}')

            # Keep only the specified columns
            y_df = y_df[y_keep_cols]

            # Convert wt sequence to array of characters
    wt_seq_a = np.array(list(wt_seq))

    # Create NxL array of characters, then ravel
    x_arr = np.tile(wt_seq_a, [N, 1]).ravel()

    # Get indices to replace
    ix = L * ids + ls

    # Replace characters
    x_arr[ix] = cs

    # Reshape to NxL array
    x_arr = x_arr.reshape([N, L])

    # Convert each row to string.
    # Note the strange magic incantation needed, since row.tostring()
    # returns a byte string instead of a unicode stirng.
    # Might be possible to speed this up.
    x = np.array(
        [str(row.tostring(), 'utf-8').replace('\x00', '') for row in x_arr])

    # Return dataframe
    data_df = pd.DataFrame()
    data_df['x'] = x
    data_df.index = range(N)

    # Merge with y_df, if provided
    if y_df is not None:
        data_df = pd.merge(left=y_df, right=data_df, left_index=True,
                           right_index=True, how='right')

    # Make it clear that data_df index are variant id numbers
    data_df.index.name = 'id'

    return data_df

