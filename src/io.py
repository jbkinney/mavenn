import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import IUPAC
import re
import pdb

def validate_dataframe(df, fix=False):
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
    if not df.shape[1] >= 1 and df.shape[0] >= 1:
        raise TypeError(\
            'Data frame must contain at least one row and one column')

    #
    # Validate sequence columns
    #
    seq_cols = [c for c in df.columns if re.match(r'^seq',c)]
    for col in seq_cols:

        # Set alphabet
        if col=='seq':
            alphabet = IUPAC.IUPACUnambiguousDNA.letters
        elif col=='seq_rna':
            alphabet = IUPAC.IUPACUnambiguousRNA.letters
        elif col=='seq_pro':
            alphabet = IUPAC.IUPACProtein.letters
        else:
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

        # Verify that counts are nonnegative
        if not all(df[col] >= 0):
            raise TypeError('Counts must be nonnegative numbers.')

    #
    # Validate parameter columns
    #
    val_cols = [c for c in df.columns if re.match(r'^val',c)]
    for col in val_cols:

        # Check if columns are floats
        if not df[col].values.dtype == float
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
        if not all(np.isfinite(df['ct']))
            TypeError('Nonfinite parameters encountered.')

    #
    # TO DO: order columns
    #

    if fix:
        return df
    else
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
        df = pd.io.parsers.read_csv(file_name,delim_whitespace=True)

    # For fastq or fasta file, use Bio.SeqIO
    elif file_type=='fastq' or file_type=='fasta':
        df = pd.DataFrame(columns=['seq'])
        for i,record in enumerate(SeqIO.parse(fn,file_type)):
            df.loc[i] = str(record.seq)

    # Return validated/fixed dataframe
    df_valid =  validate_dataframe(df, fix=True)
    return df_valid