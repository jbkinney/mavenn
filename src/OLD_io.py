import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import IUPAC
import re
import pdb
import numpy as np

# Make list of all possible parameters
alphabets = {
    'seq':''.join(sorted(IUPAC.IUPACUnambiguousDNA.letters)),
    'seq_rna':''.join(sorted(IUPAC.IUPACUnambiguousRNA.letters)),
    'seq_pro':''.join(sorted(IUPAC.IUPACProtein.letters))
}
model_columns = []
for alphabet in alphabets.values():
    cols_mat = ['pos'] + ['val_'+a for a in alphabet] 
    cols_nbr = ['pos']
    for a in alphabet:
        for b in alphabet:
            cols_nbr.append('val_'+a+b)
    model_columns.append(cols_mat)
    model_columns.append(cols_nbr)

def validate(df, fix=False):
    """ 
    Validates the content of a data fram for consistency with 
    sortseq standards. Should be run after reading files and before
    writing files, as well as any time the data frame you're working
    with has unverified content. 

    Arguments:

    fix (bool): if True, returns fixed-up dataframe if dataframe can be fixed.
        If False, raises a TypeError if any problems are found, and 
        returns True otherwise.
    """

    # Verify that the data frame has at least one row and one column
    if not df.shape[0] >= 1:
        raise TypeError(\
            'Data frame must contain at least one row')
    if not df.shape[1] >= 1:
        raise TypeError(\
            'Data frame must contain at least one column')

    #
    # Validate column names
    #
    exact_cols = ['seq', 'seq_pro', 'seq_rna', 'ct', 'pos','bin','file']
    col_patterns = [r'^ct_',r'^val_']
    for col in df.columns:
        if (not col in exact_cols) and \
            (not any([re.match(pat,col) for pat in col_patterns])):
            raise TypeError('Invalid column in dataframe: %s.'%col)

    #
    # Validate sequence columns
    #
    seq_cols = [c for c in df.columns if re.match(r'^seq',c)]
    for col in seq_cols:

        # Set alphabet
        try:
            alphabet = alphabets[col]
        except:
            raise TypeError('Sequence column is of unkown type: %s.'%col)

        # Check that all sequences have the same length
        L = len(df[col][0])
        if not all([len(seq) == L for seq in df[col]]):
            raise TypeError('Not all sequences are the same length.')

        # Make sure sequences are uppercase
        if not all([seq==seq.upper() for seq in df[col]]):
            if fix:
                df[col] = [seq.upper() for seq in df[col]]
            else:
                TypeError('Seqs are not all uppercase; set fix=True to fix.')

        # Check that all characters are from the correct alphabet
        search_string = r"[^%s]"%alphabet
        if not all([re.search(search_string,seq)==None for seq in df[col]]):
            raise TypeError('Invalid character found in sequences.')

    #
    # Validate count columns
    #
    count_cols = [c for c in df.columns if re.match(r'^ct',c)]
    for col in count_cols:

        # Verify that counts are integers
        if not df[col].values.dtype == int:

            # Try to convert column to numbers
            try:
                int_vals = df[col].astype(int)
                float_vals = df[col].astype(float)
            except:
                raise TypeError('Cannot interptret counts as integers.')

            # Convert to integers if this doesn't change count values
            if all(int_vals == float_vals):
                if fix:
                    df[col] = int_vals
                else:
                    TypeError('Counts are not integers; set fix=True to fix.')
            else:
                raise TypeError('Noninteger numbers found in counts.')

            # Make sure that all parameters are finite
            if not all(np.isfinite(df[col])):
                TypeError('Nonfinite counts encountered.')

        # Verify that counts are nonnegative
        if not all(df[col] >= 0):
            raise TypeError('Counts must be nonnegative numbers.')

    #
    # Validate parameter columns
    #
    val_cols = [c for c in df.columns if re.match(r'^val',c)]
    for col in val_cols:

        # Check if columns are floats
        if not df[col].values.dtype == float:
            try:
                float_vals = df[col].astype(float)
            except:
                raise TypeError('Cannot interptret counts as integers.')

            # Convert to floats if this doesn't change values
            if all(float_vals == df[col]):
                if fix:
                    df[col] = float_vals
                else:
                    TypeError('Parameter values are not floats; set fix=True to fix.')
            else:
                raise TypeError('Non-float parameters encountered.')

        # Make sure that all parameters are finite
        if not all(np.isfinite(df[col])):
            raise TypeError('Nonfinite parameters encountered.')

    #
    # Validate position column
    #
    col = 'pos'
    if col in df.columns:
        # Verify that positions are consecutive
        first = df[col].iloc[0]
        last = df[col].iloc[-1]
        if not all(df[col] == range(first,last+1)):
            raise TypeError('Positions are not consecutive integers.')

        # Verify that positions are nonnegative
        if first < 0:
            raise TypeError('Positions are not all nonnegative.')

        # Verify that positions are of type int 
        if not df[col].values.dtype == int:
            if fix:
                    df[col] = df[col].astype(int)
            else:
                TypeError('Counts are not integers; set fix=True to fix.')


        # Verify that positions are consecutive integer

    #
    # Validate column order
    #
    ct_cols = sorted([col for col in df.columns if re.match(r'^ct',col)])
    val_cols = sorted([col for col in df.columns if re.match(r'^val',col)])
    pos_cols = sorted([col for col in df.columns if re.match(r'pos',col)])
    seq_cols = sorted([col for col in df.columns if re.match(r'^seq',col)])
    tmp = ct_cols + val_cols + pos_cols + seq_cols
    other_cols = sorted([col for col in df.columns if not col in tmp])
    new_cols = pos_cols + ct_cols + val_cols + other_cols + seq_cols
    if not all(df.columns == new_cols):
        if fix:
            df = df[new_cols]
        else:
            raise TypeError('Dataframe columns are in the wrong order; set fix=True to fix.')

    # If dataframe represents a model, validate columns exactly
    if len(val_cols) > 0:
        ok = False
        for cols in model_columns:
            # Check if cols and df.columns are identical
            if len(cols)==len(df.columns):
                if all([a==b for a,b in zip(cols,df.columns)]):
                    ok = True
        if not ok:
            raise TypeError('Dataframe represents model with invalid columns.')

    # Return fixed df if 
    if fix:
        return df
    else:
        return True

def load(file_name, file_type='text'):
    """ Loads a file, returns a data frame. 
        Can load text, fasta, or fastq files
    """

    # Check that file type is vaild
    valid_types = ['text','fasta','fastq']
    if not file_type in valid_types:
        raise TypeError('Argument file_type, = %s, is not valid.'%\
            str(file_type))

    # For text file, just load as whitespace-delimited data frame
    if file_type=='text':
        df = pd.io.parsers.read_csv(file_name,delim_whitespace=True,\
            comment='#')

    # For fastq or fasta file, use Bio.SeqIO
    elif file_type=='fastq' or file_type=='fasta':
        df = pd.DataFrame(columns=['seq'])
        for i,record in enumerate(SeqIO.parse(file_name,file_type)):
            df.loc[i] = str(record.seq)

    # If a sequences were loaded, or tags were loaded, and there are no counts,
    # then add counts
    if any([re.match(r'^seq',col) for col in df.columns]) or \
        any([re.match(r'tag',col) for col in df.columns]):
        if not any([re.match(r'^ct',col) for col in df.columns]):
            df['ct'] = 1

    # Return validated/fixed dataframe
    df_valid = validate(df, fix=True)
    return df_valid

def write(df,file_handle):
    """ Writes a data frame to specified file, given handle
    """

    # Validate dataframe
    df_valid = validate(df, fix=True)

    # Write dataframe to file
    file_handle.write(df_valid.to_string())

