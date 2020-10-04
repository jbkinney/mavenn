"""mavedb.py: Utilities for working with data from MaveDB."""
# Standard imports
import numpy as np
import pandas as pd
import re
import tempfile
import os
import pdb

# MAVE-NN imports
from mavenn.src.error_handling import check, handle_errors
from mavenn.src.utils import x_to_alphabet
from mavenn.src.validate import abreviation_dict


@handle_errors
def mavedb_to_dataset(df,
                      hgvs_col,
                      y_col,
                      other_cols=None,
                      training_frac=.8,
                      seed=0,
                      dropna_y=True):
    """
    Convert variants dataset from MaveDB format to a MAVE-NN compatible dataset.

    Parameters
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

    Returns
    -------
    data_df: (pd.DataFrame)
        MAVE-NN compatible dataset.

    info_dict: (dict)
        Metadata pertaining to dataset.
    """
    # TODO: Write checks

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
def mutations_to_dataset(wt_seq,
                         mut_df,
                         y_df=None,
                         l_col="l",
                         c_col="c",
                         id_col="id",
                         y_id_col=None,
                         y_keep_cols=None):
    """
    Generate dataset from a wildetype sequence and list of mutations.

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

    Parameters
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

    Returns
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
