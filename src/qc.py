import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import IUPAC
import re
import pdb
import numpy as np
from sst import SortSeqError

# Set of regular expression patterns used to identify columns
col_patterns = {
    'seqs'  :   r'^seq$|^seq_rna$|^seq_pro$',
    'tag'   :   r'^tag$',
    'cts'   :   r'^ct',
    'ct_'   :   r'^ct_',
    'ct'    :   r'^ct$',
    'file'  :   r'^file$',
    'bin'   :   r'^bin$',
    'pos'   :   r'^pos$',
    'vals'  :   r'^val_',
    'infos' :   r'^info',
    'errs'  :   r'err$',
    'freq_' :   r'^freq_',
    'wts'   :   r'^wt$|^wt_rna$|^wt_pro$',
    'mut'   :   r'^mut$',
    'muts'  :   r'^mut$|^mut_err$'
}

seqtype_to_alphabet_dict = {
    'dna':''.join(sorted(IUPAC.IUPACUnambiguousDNA.letters)),
    'rna':''.join(sorted(IUPAC.IUPACUnambiguousRNA.letters)),
    'protein':''.join(sorted(IUPAC.IUPACProtein.letters+'*'))
}
alphabet_to_seqtype_dict = {v: k for k, v in seqtype_to_alphabet_dict.items()}

colname_to_seqtype_dict = {
    'seq':'dna',
    'seq_rna':'rna',
    'seq_pro':'protein',
    'tag':'dna',
    'wt':'dna',
    'wt_rna':'rna',
    'wt_pro':'protein'
}
seqtype_to_wtcolname_dict = {
    'dna':'wt',
    'rna':'wt_rna',
    'protein':'wt_pro'
}
seqtype_to_seqcolname_dict = {
    'dna':'seq',
    'rna':'seq_rna',
    'protein':'seq_pro'
}
seqtypes = ['dna','rna','protein']

# List of parameter names for different types of models 
model_parameters_dict = {}
for key in seqtype_to_alphabet_dict.keys():
    alphabet = seqtype_to_alphabet_dict[key]
    vals_mat = ['val_'+a for a in alphabet] 
    vals_nbr = []
    for a in alphabet:
        for b in alphabet:
            vals_nbr.append('val_'+a+b)
    model_parameters_dict['mat_'+key] = vals_mat
    model_parameters_dict['nbr_'+key] = vals_nbr


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
        raise SortSeqError('col_types is not a string or a list.')

    # Check for matches wihtin col_type list
    for col_type in col_types_list:
        pattern = col_patterns[col_type]
        if re.search(pattern,col_name):
            col_match = True

    # Return true if any match found
    return col_match


def get_cols_from_df(df,col_types):
    """
    Returns all colnames of a given type from a df, sorted alphabetically
    """
    return sorted([c for c in df.columns if is_col_type(c,col_types)])


def _validate_cols(df, fix=False):
    """
    Validates the contents of columns of all types in a dataframe
    """
    df = _validate_seqs_cols(df,fix=fix)
    df = _validate_cts_cols(df,fix=fix)
    df = _validate_pos_cols(df,fix=fix)
    df = _validate_bin_cols(df,fix=fix)
    df = _validate_mut_cols(df,fix=fix)
    df = _validate_val_cols(df,fix=fix)
    df = _validate_freq_cols(df,fix=fix)
    df = _validate_info_cols(df,fix=fix)
    return df


def _validate_seqs_cols(df, fix=False):
    """
    Validates sequence columns in a given dataframe. Will check columns with names seq, seq_rna, seq_pro, tag, wt, wt_rna, wt_pro
    """
    seq_cols = get_cols_from_df(df,['seqs','tag','wts'])
    for col in seq_cols:

        # Set alphabet
        try:
            seqtype = colname_to_seqtype_dict[col]
            alphabet = seqtype_to_alphabet_dict[seqtype]
        except:
            raise SortSeqError('Sequence column is of unkown type: %s.'%col)

        # Check that all sequences have the same length
        L = len(df[col][0])
        if not all([len(seq) == L for seq in df[col]]):
            raise SortSeqError('Not all sequences are the same length.')

        # Make sure sequences are uppercase
        if not all([seq==seq.upper() for seq in df[col]]):
            if fix:
                df[col] = [seq.upper() for seq in df[col]]
            else:
                SortSeqError('Seqs are not all uppercase; set fix=True to fix.')

        # Check that all characters are from the correct alphabet
        search_string = r"[^%s]"%alphabet
        if not all([re.search(search_string,seq)==None for seq in df[col]]):
            raise SortSeqError('Invalid character found in sequences.')

    return df

def _validate_cts_cols(df, fix=False):
    """
    Validates count columns in a given dataframe. Will check columns with names ct or ct_*.
    """
    count_cols = get_cols_from_df(df,'cts')
    for col in count_cols:

        # Verify that counts are integers
        if not df[col].values.dtype == int:

            # Try to convert column to numbers
            try:
                int_vals = df[col].astype(int)
                float_vals = df[col].astype(float)
            except:
                raise SortSeqError('Cannot interptret counts as numbers; column name = %s'%col)

            # Convert to integers if this doesn't change count values
            if all(int_vals == float_vals):
                if fix:
                    df[col] = int_vals
                else:
                    SortSeqError('Counts are not integers; set fix=True to fix.')
            else:
                raise SortSeqError('Noninteger numbers found in counts.')

            # Make sure that all parameters are finite
            if not all(np.isfinite(df[col])):
                SortSeqError('Nonfinite counts encountered.')

        # Verify that counts are nonnegative
        if not all(df[col] >= 0):
            raise SortSeqError('Counts must be nonnegative numbers.')

    return df


 # Validate position column

def _validate_pos_cols(df, fix=False):
    """
    Validates the pos column in a given dataframe (if it exists)
    """
    col = 'pos'
    if col in df.columns:
        try:
            int_vals = df[col].values.astype(int)
            float_vals = df[col].values.astype(float)
        except:
            raise SortSeqError(\
                'Cannot convert values in column %s to numbers.'%col)

        if not df[col].values.dtype == int:
            if all(int_vals==float_vals):
                if fix:
                    df[col] = df[col].astype(int)
                else:
                    raise SortSeqError(\
                        'Positions are not integers; set fix=True to fix.')
            else:
                raise SortSeqError(\
                        'Positions cannot be interpreted as integers.')

        first = df[col].iloc[0]
        last = df[col].iloc[-1]
        if not all(df[col] == range(first,last+1)):
            raise SortSeqError('Positions are not consecutive integers.')

        if first < 0:
            raise SortSeqError('Positions are not all nonnegative.')
        
    return df

def _validate_bin_cols(df, fix=False):
    """
    Validates the bin column in a given dataframe (if it exists)
    """
    col = 'bin'
    if col in df.columns:
        try:
            int_vals = df[col].values.astype(int)
            float_vals = df[col].values.astype(float)
        except:
            raise SortSeqError(\
                'Cannot convert values in column %s to numbers.'%col)

        if not df[col].values.dtype == int:
            if all(int_vals==float_vals):
                if fix:
                    df[col] = df[col].astype(int)
                else:
                    raise SortSeqError(\
                        'Positions are not integers; set fix=True to fix.')
            else:
                raise SortSeqError(\
                        'Positions cannot be interpreted as integers.')

        if not len(int_vals)==len(set(int_vals)):
            raise SortSeqError('Bin numbers are not unique.')

        if not all(int_vals >= 0):
            raise SortSeqError('Bin numbers must be nonnegative numbers.')

    return df

def _validate_mut_cols(df, fix=False):
    """
    Validates contents of mut and mut_err columns in a given dataframe
    """
    mut_cols = get_cols_from_df(df,'muts')
    for col in mut_cols:

        # Verify that freqs are floats
        if not df[col].values.dtype == float:

            # Check whether freqs can be interpreted as floats
            try:
                float_vals = df[col].astype(float)
                assert all(df[col] == float_vals)
            except:
                raise SortSeqError('Non-numbers found in freqs.')

            # Check whether we have permission to change these to floats
            if fix:
                df[col] = df[col].astype(float)
            else:
                raise SortSeqError(\
                    'Freqs are not floats; set fix=True to fix.')

        # Make sure that all mut values are between 0 and 1
        if col=='mut':
            if (not all(df[col]<=1.0)) or (not all(df[col]>=0.0)):
                raise SortSeqError(\
                    'Freq values outside [0.0, 1.0] encountered.')
    return df

def _validate_val_cols(df, fix=False):
    """
    Validates contents of val_* columns in a given dataframe
    """
    val_cols = get_cols_from_df(df,'vals')
    for col in val_cols:

        # Check if columns are floats
        if not df[col].values.dtype == float:
            try:
                float_vals = df[col].astype(float)
            except:
                raise SortSeqError('Cannot interptret counts as integers.')

            # Convert to floats if this doesn't change values
            if all(float_vals == df[col]):
                if fix:
                    df[col] = float_vals
                else:
                    SortSeqError('Parameter values are not floats; set fix=True to fix.')
            else:
                raise SortSeqError('Non-float parameters encountered.')

        # Make sure that all parameters are finite
        if not all(np.isfinite(df[col])):
            raise SortSeqError('Nonfinite parameters encountered.')
    return df

def _validate_freq_cols(df, fix=False):
    """
    Validates contents of freq_* columns in a given dataframe
    """
    freq_cols = get_cols_from_df(df,'freq_')
    for col in freq_cols:

        # Verify that freqs are floats
        if not df[col].values.dtype == float:

            # Check whether freqs can be interpreted as floats
            try:
                float_vals = df[col].astype(float)
                assert all(df[col] == float_vals)
            except:
                raise SortSeqError('Non-numbers found in freqs.')

            # Check whether we have permission to change these to floats
            if fix:
                df[col] = df[col].astype(float)
            else:
                raise SortSeqError(\
                    'Freqs are not floats; set fix=True to fix.')

        # Make sure that all freqs are between 0 and 1
        if (not all(df[col]<=1.0)) or (not all(df[col]>=0.0)):
            raise SortSeqError('Freq values outside [0.0, 1.0] encountered.')

    return df

def _validate_info_cols(df, fix=False):
    """
    Validates contents of info and info_err columns in a given dataframe
    """
    info_cols = get_cols_from_df(df,'infos')
    for col in info_cols:

        # Verify that freqs are floats
        if not df[col].values.dtype == float:

            # Check whether freqs can be interpreted as floats
            try:
                float_vals = df[col].astype(float)
                assert all(df[col] == float_vals)
            except:
                raise SortSeqError('Non-numbers found in freqs.')

            # Check whether we have permission to change these to floats
            if fix:
                df[col] = df[col].astype(float)
            else:
                SortSeqError(\
                    'Info values are not floats; set fix=True to fix.')
    return df

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
        raise SortSeqError('Dataframe must contain at least one row')

    # Validate column names
    for col in df.columns:
        if not is_col_type(col,['seqs','cts','tag']):
            raise SortSeqError('Invalid column in dataframe: %s'%col)

    # Validate contents of columns
    df = _validate_cols(df,fix=fix)

    # Validate column order
    ct_cols = get_cols_from_df(df,'cts')
    tag_cols = get_cols_from_df(df,'tag')
    seq_cols = get_cols_from_df(df,'seqs')
    new_cols = ct_cols + tag_cols + seq_cols
    if not all(df.columns == new_cols):
        if fix:
            df = df[new_cols]
        else:
            raise SortSeqError('Dataframe columns are in the wrong order; set fix=True to fix.')

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

    # Verify dataframe has at least one row and one column
    if not df.shape[0] >= 1:
        raise SortSeqError(\
            'Dataframe must contain at least one row')

    # Check for exactly one tag column
    tag_cols = get_cols_from_df(df,'tag')
    if len(tag_cols) != 1:
        raise SortSeqError('Must be exactly one tag column.')

    # Check for exactly one seqs column
    seq_cols = get_cols_from_df(df,'seqs')
    if len(seq_cols) != 1:
        raise SortSeqError('Must be exactly one sequence column.')

    # Validate contents of columns
    df = _validate_cols(df,fix=fix)

    # Rearrange columns
    new_cols = tag_cols + seq_cols
    if not all(df.columns==new_cols):
        if fix:
            df = df[new_cols]
        else:
            raise SortSeqError('Dataframe columns are in the wrong order; set fix=True to fix.')

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
        raise SortSeqError(\
            'Dataframe must contain at least one row')

    # Verify dataframe has exactly two columns
    if not df.shape[1] == 2:
        raise SortSeqError(\
            'Dataframe must contain exactly two columns')

    # Verify dataframe has exactly two columns with the proper names
    exact_cols = ['bin', 'file']
    for col in df.columns:
        if not is_col_type(col,exact_cols):
            raise SortSeqError('Invalid column in dataframe: %s.'%col)
    for col in exact_cols:
        if not col in df.columns:
            raise SortSeqError('Could not find column in dataframe: %s'%col)

    # Fix column order if necessary
    if not all(df.columns == exact_cols):
        if fix:
            df = df[exact_cols]
        else:
            raise SortSeqError('Dataframe columns are in the wrong order; set fix=True to fix.')

    # Validate contents of columns
    df = _validate_cols(df,fix=fix)

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
        raise SortSeqError(\
            'Dataframe must contain at least one row')

    # Validate column names
    for col in df.columns:
        if not is_col_type(col,['pos','vals']):
            raise SortSeqError('Invalid column in dataframe: %s.'%col)

    # Validate parameter column names
    val_cols = sorted([c for c in df.columns if is_col_type(c,'vals')])
    ok = False
    for cols in model_parameters_dict.values():
        # Check if cols and df.columns are identical
        if len(cols)==len(val_cols):
            if all([a==b for a,b in zip(cols,val_cols)]):
                ok = True
    if not ok:
        raise SortSeqError('Dataframe represents model with invalid columns: %s'%str(val_cols))

    # Validate contents of all columns
    df = _validate_cols(df,fix=fix)

    return df


# Validates profile_ct dataframes
def validate_profile_ct(df, fix=False):
    """ 
    Validates the form of a profile_ct dataframe. A profile_ct dataframe must look something like this:

    pos     ct      ct_A    ct_C    ct_G    ct_T
    0       5       1       2       0       2
    1       5       2       0       0       3
    2       5       0       0       1       4

    Specifications:
    0. The dataframe must have at least one row.
    1. A 'pos' column is mandatory and should appear first. Values must be sequential nonnegative integers.
    2. A set of 'ct_X' columns is mandatory; these must correpsond to a valid alphabet. Counts must be nonnegative integers. 
    3. A 'ct' column, which lists the sum of counts for each character in each row. Values must be the same in all positions. 

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
        raise SortSeqError(\
            'Dataframe must contain at least one row')

    # Validate column names
    for col in df.columns:
        if not is_col_type(col,['pos','cts']):
            raise SortSeqError('Invalid column in dataframe: %s.'%col)

    # Validate contents of all columns
    df = _validate_cols(df,fix=fix)

    # Validate that column 'ct' contains sum; fix if this is missing
    total_ct_col = 'ct'
    char_ct_cols = get_cols_from_df(df,'ct_')
    total_counts = df[char_ct_cols].sum(axis=1)
    if not total_ct_col in df.columns:
        if fix:
            df[total_ct_col] = total_counts
        else:
            raise SortSeqError('"ct" column not found; set fix=True to fix.')
    else:
        if not all(total_counts == df[total_ct_col]):
            raise SortSeqError('"ct_X" columns do not sum to "ct" column.')

    # Validate the 'ct' value is the same at all positions
    if len(set(total_counts.values))>1:
        raise SortSeqError('"ct" value must be same at all pos.')

    # Validate column order
    ct_cols = get_cols_from_df(df,'cts')
    pos_cols = get_cols_from_df(df,'pos')
    new_cols = pos_cols + ct_cols
    if not all(df.columns == new_cols):
        if fix:
            df = df[new_cols]
        else:
            raise SortSeqError('Dataframe columns are in the wrong order; set fix=True to fix.')

    return df


# Validates profile_freq dataframes
def validate_profile_freq(df, fix=False):
    """ 
    Validates the form of a profile_freq dataframe. A profile_freq dataframe must look something like this:

    pos     freq_A  freq_C  freq_G  freq_T
    0       0.3     0.2     0.1     0.4
    1       0.4     0.3     0.2     0.1
    2       0.2     0.1     0.4     0.3

    Specifications:
    0. The dataframe must have at least one row.
    1. A 'pos' column is mandatory and should appear first. Values must be sequential nonnegative integers.
    2. A set of 'freq_X' columns is mandatory; these must correpsond to a valid alphabet. Values must be floats between 0.0 and 1.0, inclusive.

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
        raise SortSeqError(\
            'Dataframe must contain at least one row')

    # Validate column names
    for col in df.columns:
        if not is_col_type(col,['pos','freq_']):
            raise SortSeqError('Invalid column in dataframe: %s.'%col)

    df = _validate_cols(df,fix=fix)

    # Validate column order
    freq_cols = get_cols_from_df(df,'freq_')
    new_cols = ['pos'] + freq_cols
    if not all(df.columns == new_cols):
        if fix:
            df = df[new_cols]
        else:
            raise SortSeqError('Dataframe columns are in the wrong order; set fix=True to fix.')

    return df


# Validates profile_mut dataframes
def validate_profile_mut(df, fix=False):
    """ 
    Validates the form of a profile_mut dataframe. A profile_mut dataframe  looks something like this:

    pos     wt    mut   mut_err
    0       A     0.23      0.1
    1       C     0.20      0.1
    2       G     0.26      0.2

    Specifications:
    0. The dataframe must have at least one row.
    1. A 'pos' column is mandatory and should appear first. Values must be sequential nonnegative integers.
    2. A 'wt' column must appear next. 'wt' corresponds to DNA, 'wt_rna' corresponds to RNA, and 'wt_pro' corresponds to protein
    3. A 'mut' column must appear next. Values are between 0.0 and 1.0 inclusive.
    4. A 'mut_err' column comes next but is optional. Values must be nonnegative numbers.

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
        raise SortSeqError(\
            'Dataframe must contain at least one row')

    # Validate column names
    for col in df.columns:
        if not is_col_type(col,['pos','wts','muts']):
            raise SortSeqError('Invalid column in dataframe: %s.'%col)

    # Get wt columns
    wt_cols = get_cols_from_df(df,'wts')
    if not len(wt_cols)==1:
        SortSeqError('Multiple wt columns found.')

    # Get mut cols
    mut_cols = get_cols_from_df(df,'muts')
    
    # Validate contents of all columns
    df = _validate_cols(df,fix=fix)

    # Validate column order
    new_cols = ['pos'] + wt_cols + mut_cols
    if not all(df.columns == new_cols):
        if fix:
            df = df[new_cols]
        else:
            raise SortSeqError('Dataframe columns are in the wrong order; set fix=True to fix.')

    return df



# Validates information profile dataframes
def validate_profile_info(df, fix=False):
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

    # Verify dataframe has at least one row
    if not df.shape[0] >= 1:
        raise SortSeqError(\
            'Dataframe must contain at least one row')

    # Validate column names
    for col in df.columns:
        if not is_col_type(col,['pos','infos']):
            raise SortSeqError('Invalid column in dataframe: %s.'%col)

    # Validate contents of columns
    df = _validate_cols(df,fix=fix)

    # Make sure that all info values are nonnegative
    info_cols = get_cols_from_df(df,'infos')
    if not 'info' in info_cols:
        raise SortSeqError('info column is missing.')

    # Validate column order
    new_cols = ['pos'] + info_cols
    if not all(df.columns == new_cols):
        if fix:
            df = df[new_cols]
        else:
            raise SortSeqError(\
             'Dataframe columns are in the wrong order; set fix=True to fix.')

    return df

