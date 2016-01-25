import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import IUPAC
import re
import pdb
import numpy as np
import os
import qc
import utils
from sst import SortSeqError

def validate_file_for_reading(file_arg):
    """ Checks that a specified file exists and is readable. Returns a valid file handle given a file name or handle 
    """
    # If user passed file name
    if type(file_arg)==str:

        # Verify that file exists
        if not os.path.isfile(file_arg):
            raise SortSeqError('Cannot find file: %s'%file_arg)

        # Verify that file can be read
        if not os.access(file_arg,os.R_OK):
            raise SortSeqError('Can find but cannot read from file: %s'%file_arg)

        # Get handle to file
        file_handle = open(file_arg,'r')

    # If user passed file object
    elif type(file_arg)==file:

        # Verify that file isn't closed
        if file_arg.closed:
            raise SortSeqError('File object is already closed.')
        file_handle = file_arg

    # Otherwise, throw error
    else:
        raise SortSeqError('file_arg is neigher a name or handle.')

    # Return validated file handle
    return file_handle


def validate_file_for_wrtiting(file_arg):
    """ Checks that a specified file can be written
    """
    # If user passed file name
    if type(file_arg)==str:

        # Get handle to file
        file_handle = open(file_arg,'w')

        # Verify that file can be read
        if not os.access(file_arg,os.W_OK):
            raise SortSeqError('Cannot write to file: %s'%file_arg)

    # If user passed file object
    elif type(file_arg)==file:

        file_handle = file_arg

    # Otherwise, throw error
    else:
        raise SortSeqError('file_arg is neigher a name or handle.')

    # Return validated file handle
    return file_handle


def load_dataset(file_arg, file_type='text',seq_type=None):
    """ Loads a dataset file, returns a data frame. 
        Can load text, fasta, or fastq files
    """
    # Check that the file can be read
    file_handle = validate_file_for_reading(file_arg)

    # Check that file type is vaild
    valid_types = ['text','fasta','fastq','raw']
    if not file_type in valid_types:
        raise SortSeqError('Argument file_type, = %s, is not valid.'%\
            str(file_type))

    # If seq_type is specified, get correposnding colname
    if seq_type:
        if not seq_type in qc.seqtype_to_seqcolname_dict.keys():
            raise SortSeqError('seq_type %s is invalid'%str(seq_type))
        colname = qc.seqtype_to_seqcolname_dict[seq_type]

    # For text file, just load as whitespace-delimited data frame
    if file_type=='text':
        df = pd.read_csv(file_arg,delim_whitespace=True,\
            comment='#')

        # If seq_type is specified, make sure it matches colname in df
        if seq_type and (not colname in df.columns):
            raise SortSeqError('Column %s is not in dataframe'%str(colname))

    # For raw text, load into dataframe with column defined by seq_type
    elif file_type=='raw':
        if not seq_type:
            raise SortSeqError('file_type=="raw" but seq_type is not set.')

        df = pd.read_csv(file_arg,delim_whitespace=True,\
            comment='#', header=None)
        if len(df.columns)!=1:
            raise SortSeqError(\
                'file_type=="raw" but file has multiple columns.')
        df.columns=[colname]

    # For fastq or fasta file, use Bio.SeqIO
    elif file_type=='fastq' or file_type=='fasta':

        # Raise error if seq_type was not specified
        if not seq_type:
            raise SortSeqError(\
                'Seqtype unspecified while fasta or fastq file.')

        # Fill in dataframe with fasta or fastq data
        df = pd.DataFrame(columns=[colname])
        for i,record in enumerate(SeqIO.parse(file_handle,file_type)):
            df.loc[i] = str(record.seq)

    # If a sequences or tags were loaded, 
    # and there are no counts, then add counts
    if any([qc.is_col_type(col,['seqs','tag']) for col in df.columns]):
        if not any([qc.is_col_type(col,'cts') for col in df.columns]):
            df['ct'] = 1

    # Return validated/fixed dataframe
    df_valid = qc.validate_dataset(df, fix=True)
    return df_valid


def load_model(file_arg):
    """ Loads a model from a text file into a dataframe
    """

    # Check that the file can be read
    file_handle = validate_file_for_reading(file_arg)

    df = pd.read_csv(file_handle,delim_whitespace=True,\
            comment='#')

    # Return validated/fixed dataframe
    df_valid = qc.validate_model(df, fix=True)
    return df_valid


def load_filelist(file_arg):
    """ Loads a filelist from a text file into a dataframe
    """

    # Check that the file can be read
    file_handle = validate_file_for_reading(file_arg)

    df = pd.read_csv(file_handle,delim_whitespace=True,comment='#')

    # Return validated/fixed dataframe
    df_valid = qc.validate_filelist(df, fix=True)
    return df_valid


def load_tagkey(file_arg):
    """ Loads a tagkey from a text file into a dataframe
    """

    # Check that the file can be read
    file_handle = validate_file_for_reading(file_arg)

    df = pd.read_csv(file_handle,delim_whitespace=True,\
            comment='#')

    # Return validated/fixed dataframe
    df_valid = qc.validate_tagkey(df, fix=True)
    return df_valid



def load_profile_ct(file_arg):
    """ Loads a profile_ct dataframe from a text file
    """

    # Check that the file can be read
    file_handle = validate_file_for_reading(file_arg)

    df = pd.read_csv(file_handle,delim_whitespace=True,\
            comment='#')

    # Return validated/fixed dataframe
    df_valid = qc.validate_profile_ct(df, fix=True)
    return df_valid


def load_profile_freq(file_arg):
    """ Loads a profile_freq dataframe from a text file
    """

    # Check that the file can be read
    file_handle = validate_file_for_reading(file_arg)

    df = pd.read_csv(file_handle,delim_whitespace=True,\
            comment='#')

    # Return validated/fixed dataframe
    df_valid = qc.validate_profile_freq(df, fix=True)
    return df_valid


def load_profile_mut(file_arg):
    """ Loads a profile_mut dataframe from a text file
    """

    # Check that the file can be read
    file_handle = validate_file_for_reading(file_arg)

    df = pd.read_csv(file_handle,delim_whitespace=True,\
            comment='#')

    # Return validated/fixed dataframe
    df_valid = qc.validate_profile_mut(df, fix=True)
    return df_valid

def load_profile_info(file_arg):
    """ Loads a profile_info dataframe from a text file
    """

    # Check that the file can be read
    file_handle = validate_file_for_reading(file_arg)

    df = pd.read_csv(file_handle,delim_whitespace=True,\
            comment='#')

    # Return validated/fixed dataframe
    df_valid = qc.validate_profile_info(df, fix=True)
    return df_valid


def write(df,file_arg):
    """ Writes a data frame to specified file, given as name or handle
    """

    file_handle = validate_file_for_wrtiting(file_arg)

    # Write dataframe to file
    pd.set_option('max_colwidth',int(1e6)) # Dont truncate columns
    df_string = df.to_string(\
        index=False, col_space=5, float_format=utils.format_string)
    file_handle.write(df_string+'\n') # Add trailing return
    file_handle.close()
    #df.to_csv(file_arg,sep='\t')

