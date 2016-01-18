import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import IUPAC
import re
import pdb
import numpy as np

import qc

def load_dataset(file_name, file_type='text'):
    """ Loads a dataset file, returns a data frame. 
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

    # If a sequences or tags were loaded, 
    # and there are no counts, then add counts
    if any([qc.is_col_type(col,['seqs','tag']) for col in df.columns]):
        if not any([qc.is_col_type(col,'cts') for col in df.columns]):
            df['ct'] = 1

    # Return validated/fixed dataframe
    df_valid = qc.validate_dataset(df, fix=True)
    return df_valid


def load_model(file_name):
    """ Loads a model from a text file into a dataframe
    """

    df = pd.io.parsers.read_csv(file_name,delim_whitespace=True,\
            comment='#')

    # Return validated/fixed dataframe
    df_valid = qc.validate_model(df, fix=True)
    return df_valid


def load_filelist(file_name):
    """ Loads a filelist from a text file into a dataframe
    """

    df = pd.io.parsers.read_csv(file_name,delim_whitespace=True,\
            comment='#')

    # Return validated/fixed dataframe
    df_valid = qc.validate_filelist(df, fix=True)
    return df_valid


def load_tagkey(file_name):
    """ Loads a tagkey from a text file into a dataframe
    """

    df = pd.io.parsers.read_csv(file_name,delim_whitespace=True,\
            comment='#')

    # Return validated/fixed dataframe
    df_valid = qc.validate_tagkey(df, fix=True)
    return df_valid


# def write(df,file_handle):
#     """ Writes a data frame to specified file, given handle
#     """

#     # Validate dataframe
#     df_valid = validate(df, fix=True)

#     # Write dataframe to file
#     file_handle.write(df_valid.to_string())

