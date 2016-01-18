import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import IUPAC
import re
import pdb
import numpy as np

# Set of regular expression patterns used to identify columns
col_patterns = {
    'seqs':r'^seq',
    'tag':r'^tag$',
    'cts':r'^ct',
    'ct_':r'^ct_',
    'ct':r'^ct$',
    'file':r'^file$',
    'bin':r'^bin$',
    'pos':r'^pos$',
    'vals':r'^val_',
    'info':r'^info$',
    'err':r'^err$',
}

def is_col_type(col_name,col_types='all'):
    """ 
    Checks whether col_name is a valid column name, as specified by col_types. col_types can be either a string (for a single column type) or a list of strings (for multimple column types). Default col_types='all' causes function to check all available column types
    """
    col_match = False

    # Make col_types_list
    if type(col_types)==list:
        col_types_list = col_types
    elif type(col_types)==str:
        if col_types=='all':
            col_types_list = col_patterns.values()
        else:
            col_types_list = [col_types]
    else:
        raise TypeError('col_types is not a string or a list.')

    # Check for matches wihtin col_type list
    for col_type in col_types_list:
        pattern = col_patterns[col_type]
        if re.search(pattern,col_name):
            col_match = True

    # Return true if any match found
    return col_match

# Dictionary containing the letters (as a single string) of all sequence types
seq_alphabets_dict = {
    'seq':''.join(sorted(IUPAC.IUPACUnambiguousDNA.letters)),
    'seq_rna':''.join(sorted(IUPAC.IUPACUnambiguousRNA.letters)),
    'seq_pro':''.join(sorted(IUPAC.IUPACProtein.letters))
}

# List of parameter names for different types of models 
model_parameters_dict = {}
for key in seq_alphabets_dict.keys():
    alphabet = seq_alphabets_dict[key]
    vals_mat = ['val_'+a for a in alphabet] 
    vals_nbr = []
    for a in alphabet:
        for b in alphabet:
            vals_nbr.append('val_'+a+b)
    model_parameters_dict['mat_'+key] = vals_mat
    model_parameters_dict['nbr_'+key] = vals_nbr

# Validates dataset dataframes
def validate_dataset(df, fix=False):
    """ 
    Validates the form of a dataset dataframe. A dataset dataframe must look something like this:

    ct      ct_0    ct_1    ct_2    tag     seq
    3       1       2       0       CTG     ACCAT
    2       2       0       0       CTA     ACCAT
    1       0       0       1       CCA     TCAGG
    
    A 'ct' column reports the total counts of all sequence/tag pairs. Optional 'ct_0', 'ct_1', ... columns contain counts of sequence/tag. pairs for  individual bins. Optional 'tag' column lists DNA sequnce tags used to identify sequences. A 'seq' column lists the sequences of interests. 

    Specifications:
    0. The dataframe must have at least one row and one column.
    1. A 'ct' column is mandatory and should appear first. Counts must be nonnegative integers. If not present, this can be added
    2. 'ct_X' columns are optional. If they appear, X must be a nonnegative integer. Columns must appear in the order of this number. Counts must be nonnegative integers and must sum to the value in the 'ct' column. 
    3. A 'tag', 'seq', 'seq_rna', or 'seq_pro' column is mandatory. More than one of these columns are allowed simultaneously. They must appear to the left of all other columns. In each column, sequences must conform to unambiguous DNA, RNA, or protein alphabets and must be all be of the same length.

    Arguments:
        df (pd.DataFrame): Dataset in dataframe format
        fix (bool): A flag saying whether to fix the dataframe into shape if possible.

    Returns:
        if fix=True:
            df_valid: a valid dataframe that has been fixed by the function
        if fix=False:
            Nothing

    Function:
        Raises a TyepError if the data frame violates the specifications (if fix=False) or if these violations cannot be fixed (fix=True).
    """

    # Verify dataframe has at least one row and one column
    if not df.shape[0] >= 1:
        raise TypeError(\
            'Dataframe must contain at least one row')
    if not df.shape[1] >= 1:
        raise TypeError(\
            'Dataframe must contain at least one column')

    # Validate column names
    for col in df.columns:
        if not is_col_type(col,['seqs','cts']):
            raise TypeError('Invalid column in dataframe: %s.'%col)

    # Validate sequence columns
    seq_cols = [c for c in df.columns if is_col_type(c,'seqs')]
    for col in seq_cols:

        # Set alphabet
        try:
            alphabet = seq_alphabets_dict[col]
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

    # Validate count columns
    count_cols = [c for c in df.columns if is_col_type(c,'cts')]
    for col in count_cols:

        # Verify that counts are integers
        if not df[col].values.dtype == int:

            # Try to convert column to numbers
            try:
                int_vals = df[col].astype(int)
                float_vals = df[col].astype(float)
            except:
                raise TypeError('Cannot interptret counts as integers; column name = %s'%col)

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

    # Validate column order
    ct_cols = sorted([col for col in df.columns if is_col_type(col,'cts')])
    tag_cols = sorted([col for col in df.columns if is_col_type(col,'tag')])
    seq_cols = sorted([col for col in df.columns if is_col_type(col,'seqs')])
    new_cols = ct_cols + tag_cols + seq_cols
    if not all(df.columns == new_cols):
        if fix:
            df = df[new_cols]
        else:
            raise TypeError('Dataframe columns are in the wrong order; set fix=True to fix.')

    # Return fixed df if fix=True
    if fix:
        return df

# Validates model dataframes
def validate_model(df, fix=False):
    """ 
    Validates the form of a model dataframe. A model dataframe must look something like this:

    pos     val_A   val_C   val_G   val_T   
    3       1.1     4.3     -6.19   5.2
    4       0.01    3.40    -10.5   5.3
    5       0       1.4     10.9    231.0
    
    A 'pos' column reports the position within a sequence to which this modle applies. 'val_X' then describe the values of the model parameters.

    Specifications:
    0. The dataframe must have at least one row and one column.
    1. A 'pos' column is mandatory and must occur first. Values must be nonnegative integers in sequential order.
    2. 'val_X' columns must conform to one of the accepted model types. These columns must be arranged in alphabetical order. Parameter values must be finite float values.   

    Arguments:
        df (pd.DataFrame): Dataset in dataframe format
        fix (bool): A flag saying whether to fix the dataframe into shape if possible.

    Returns:
        if fix=True:
            df_valid: a valid dataframe that has been fixed by the function
        if fix=False:
            Nothing

    Function:
        Raises a TyepError if the data frame violates the specifications (if fix=False) or if these violations cannot be fixed (fix=True).
    """

    # Verify dataframe has at least one row and one column
    if not df.shape[0] >= 1:
        raise TypeError(\
            'Dataframe must contain at least one row')
    if not df.shape[1] >= 1:
        raise TypeError(\
            'Dataframe must contain at least one column')

    # Validate column names
    for col in df.columns:
        if not is_col_type(col,['pos','vals']):
            raise TypeError('Invalid column in dataframe: %s.'%col)

    # Validate position column
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

    # Validate parameter column names
    val_cols = sorted([c for c in df.columns if is_col_type(c,'vals')])
    ok = False
    for cols in model_parameters_dict.values():
        # Check if cols and df.columns are identical
        if len(cols)==len(val_cols):
            if all([a==b for a,b in zip(cols,val_cols)]):
                ok = True
    if not ok:
        raise TypeError('Dataframe represents model with invalid columns: %s'%str(val_cols))

    # Validate parameter values
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

    # Return fixed df if fix=True
    if fix:
        return df

# Validates filelist dataframes
def validate_filelist(df, fix=False):
    """ 
    Validates the form of a filelist dataframe. A filelist dataframe must look something like this:

    bin     file
    0       library.txt
    1       sample_1.txt
    2       sample_2.txt
    
    A 'bin' column reports the bin number. 'file' lists a text, fasta, or fastq file containing the sequences within each bin.

    Specifications:
    0. The dataframe must have at least one row and one column.
    1. A 'bin' column is mandatory and must occur first. Values must be unique nonnegative integers.
    2. A 'file' column is mandatory and must occur last.   

    Arguments:
        df (pd.DataFrame): Dataset in dataframe format
        fix (bool): A flag saying whether to fix the dataframe into shape if possible.

    Returns:
        if fix=True:
            df_valid: a valid dataframe that has been fixed by the function
        if fix=False:
            Nothing

    Function:
        Raises a TyepError if the data frame violates the specifications (if fix=False) or if these violations cannot be fixed (fix=True).
    """

    # Verify dataframe has at least one row
    if not df.shape[0] >= 1:
        raise TypeError(\
            'Dataframe must contain at least one row')

    # Verify dataframe has exactly two columns
    if not df.shape[1] == 2:
        raise TypeError(\
            'Dataframe must contain exactly two columns')

    # Verify dataframe has exactly two columns with the proper names
    exact_cols = ['bin', 'file']
    for col in df.columns:
        if not is_col_type(col,exact_cols):
            raise TypeError('Invalid column in dataframe: %s.'%col)
    for col in exact_cols:
        if not col in df.columns:
            raise TypeError('Could not find column in dataframe: %s'%col)

    # Fix column order if necessary
    if not all(df.columns == exact_cols):
        if fix:
            df = df[exact_cols]
        else:
            raise TypeError('Dataframe columns are in the wrong order; set fix=True to fix.')

    # Verify that all bins are unique positive integers
    col = 'bin'
    if not df[col].values.dtype == int:

        # Try to convert column to numbers
        try:
            int_vals = df[col].astype(int)
            float_vals = df[col].astype(float)
        except:
            raise TypeError('Cannot interptret bins as integers.')

        # Convert to integers if this doesn't change count values
        if all(int_vals == float_vals):
            if fix:
                df[col] = int_vals
            else:
                TypeError('Bins are not integers; set fix=True to fix.')
        else:
            raise TypeError('Noninteger bins found.')

    # Verify that bin numbers are nonnegative
    if not all(df[col] >= 0):
        raise TypeError('Bin numbers must be nonnegative numbers.')

    # Verify that bin numbers are unique
    if not len(df[col])==len(set(df[col])):
        raise TypeError('Bin numbers must be unique.')

    # Return fixed df if fix=True
    if fix:
        return df


# Validates tagkeys dataframes
def validate_tagkey(df, fix=False):
    """ 
    Validates the form of a tagkeys dataframe. A tagkeys dataframe must look something like this:

    tag     seq
    AACT    ATTAGTCTAGATC
    AGCT    ATTAGTCTAGATC
    TCGA    ATTAGTCTGGGTC
    
    A 'tag' column reports the short tag associated with the sequences in the 'seq' column. This file is used in the preprocess method

    Specifications:
    0. The dataframe must have at least one row.
    1. A 'tag' column is mandatory and must occur first. Values must be valid DNA sequences, all the same length.
    2. A single 'seq', 'seq_rna', or 'seq_pro' column is mandatory and must come second. Values must be valid DNA, RNA, or protein strings, all of the same length. 

    Arguments:
        df (pd.DataFrame): Dataset in dataframe format
        fix (bool): A flag saying whether to fix the dataframe into shape if possible.

    Returns:
        if fix=True:
            df_valid: a valid dataframe that has been fixed by the function
        if fix=False:
            Nothing

    Function:
        Raises a TyepError if the data frame violates the specifications (if fix=False) or if these violations cannot be fixed (fix=True).
    """

    #
    # Validate tag columns
    #
    tag_cols = [c for c in df.columns if is_col_type(c,'tag')]
    if len(tag_cols) != 1:
        raise TypeError('Must be exactly one tag column.'%col)
    col = 'tag'

    # Check that all tags have the same length
    L = len(df[col][0])
    if not all([len(tag) == L for tag in df[col]]):
        raise TypeError('Not all tags are the same length.')

    # Make sure tags are uppercase
    if not all([tag==tag.upper() for tag in df[col]]):
        if fix:
            df[col] = [tag.upper() for tag in df[col]]
        else:
            TypeError('Tags are not all uppercase; set fix=True to fix.')

    # Check that all characters are from the correct alphabet
    alphabet = seq_alphabets_dict['seq']
    search_string = r"[^%s]"%alphabet
    if not all([re.search(search_string,tag)==None for tag in df[col]]):
        raise TypeError('Invalid character found in tags.')

    #
    # Validate sequence columns
    #
    seq_cols = [c for c in df.columns if is_col_type(c,'seqs')]
    if len(seq_cols) != 1:
        raise TypeError('Must be exactly one sequence column.'%col)
    col = seq_cols[0]

    # Set alphabet
    try:
        alphabet = seq_alphabets_dict[col]
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
    # Rearrange columns
    #
    new_cols = tag_cols + seq_cols
    if not all(df.columns==new_cols):
        if fix:
            df = df[new_cols]
        else:
            raise TypeError('Dataframe columns are in the wrong order; set fix=True to fix.')

    # Return fixed df if fix=True
    if fix:
        return df


# Validates information profile dataframes
def validate_infoprofile(df, fix=False):
    """ 
    Validates the form of an information profile dataframe. An information profile dataframe must look something like this:

    pos     info    info_err  
    0       0.01    0.005
    1       0.03    0.006
    2       0.006   0.008
    
    A 'pos' column reports the position within a sequence to which the information profiel applies. The 'info' column describes the information in bits. The 'info_err' column quantifies uncertainty in this mutual information value. 

    Specifications:
    0. The dataframe must have at least one row and one column.
    1. A 'pos' column is mandatory and must occur first. Values must be nonnegative integers in sequential order.
    2. An 'info' column is mandatry and must come second. Values must be finite floatingpoint values. 
    3. An 'info_err' column is optional and must come last. Values must be finite floating point values. 

    Arguments:
        df (pd.DataFrame): Dataset in dataframe format
        fix (bool): A flag saying whether to fix the dataframe into shape if possible.

    Returns:
        if fix=True:
            df_valid: a valid dataframe that has been fixed by the function
        if fix=False:
            Nothing

    Function:
        Raises a TyepError if the data frame violates the specifications (if fix=False) or if these violations cannot be fixed (fix=True).
    """

    ### HAVE YET TO WRITE THIS
