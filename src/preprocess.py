#!/usr/bin/env python

'''A scripts which accepts a dataframe of input files for each bin and outputs
    a single data frame containing all the info. Combining of paired end reads,
    and any quality score filtering must occur before this script runs.'''
from __future__ import division
#Our standard Modules
import argparse
import numpy as np
import sys
import pandas as pd
import sst.utils as utils
from Bio import SeqIO
import sst.io as io
import sst.qc as qc
import re
import pdb

fasta_filename_patterns=r'(.fasta$|.fas$|.fsa$|.ffn$|.fna$|.fa$)'
fastq_filename_patterns=r'(.fastq$|.fq$)'

def merge_datasets(dataset_df_dict):
    """
    Merges multiple datasets into one. Data from disparate files is merged via values in 'tag', seq', 'seq_rna', or 'seq_pro' columns (in order of preference, chosen according to availability). Each value in the 'ct' column of each dataset is recorded in the 'ct_[bin]' column of the final dataset. A total 'ct' column is then computed, and rows in the final dataset are sorted in descending order according to this. 

    Arguments:
        dataset_df_dict (dict): Keys are bin numbers, values are dataset dataframes

    Returns:
        out_df (pd.DataFrame): A validated dataset dataframe
    """
    # Make sure datasets were loaded
    if not len(dataset_df_dict)>=1:
        raise TypeError('No datasets were loaded')

    # Determine index column. Must be same for all files
    df = dataset_df_dict.values()[0]
    if 'tag' in df.columns:
        index_col = 'tag'
    elif 'seq' in df.columns:
        index_col = 'seq'
    elif 'seq_rna' in df.columns:
        index_col = 'seq_rna'
    elif 'seq_pro' in df.columns:
        index_col = 'seq_pro'

    # Concatenate dataset dataframes
    out_df = pd.DataFrame()
    for b in dataset_df_dict.keys():
        df = dataset_df_dict[b]

        # Verify that dataframe has correct column
        if not index_col in df.columns:
            raise TypeError('\
                Dataframe does not contain index_col="%s"'%index_col)
        if not 'ct' in df.columns:
            raise TypeError('\
                Dataframe does not contain a "ct" column')

        # Delete "ct_X" columns
        for col in df.columns:
            if qc.is_col_type(col,'ct_'):
                del df[col]

        # Add bin number to name of counts column. 
        df = df.rename(columns={'ct':'ct_%d'%b})

        # Index dataset by index_col 
        df = df.groupby(index_col).sum()

        # Concatenate 
        out_df = pd.concat([out_df,df],axis=1)

    # Rename index as tag
    out_df.reset_index(inplace=True)
    out_df.rename(columns = {'index':index_col},inplace=True) 

    # Fill undefined counts with zero
    out_df.fillna(value=0,inplace=True)

    # Add 'ct' column, with proper counts
    out_df['ct'] = 0
    for col in out_df.columns:
        if qc.is_col_type(col,'ct_'):
            out_df['ct'] += out_df[col]

    # Sort by 'ct' column
    out_df.sort('ct',ascending=False,inplace=True) 
    out_df.reset_index(drop=True,inplace=True)

    # Validate out_df as dataset and return it
    out_df = qc.validate_dataset(out_df,fix=True)
    return out_df


# This is the main function, callable by the user
def main(filelist_df,tags_df=None):
    """ Merges datasets listed in the filelist_df dataframe
    """

    # Validate filelist
    qc.validate_filelist(filelist_df)

    # Read datasets into dictionary indexed by bin number
    dataset_df_dict = {}
    for item in filelist_df.iterrows():
        # Autodetect fasta, fastq, or text file based on file extension
        fn = item[1]['file']
        b = item[1]['bin']
        if re.search(fasta_filename_patterns,fn):
            df = io.load_dataset(fn,file_type='fasta')
        elif re.search(fastq_filename_patterns,fn):
            df = io.load_dataset(fn,file_type='fastq')
        else:
            df = io.load_dataset(fn,file_type='text')
        dataset_df_dict[b] = df

    # Merge datasets into one, validate, and return
    out_df = merge_datasets(dataset_df_dict)
    qc.validate_dataset(out_df)
    return out_df


# Define commandline wrapper
def wrapper(args):
    """ Commandline wrapper for main()
    """  
    
    # Run funciton
    if args.i:
        filelist_df = io.load_filelist(args.i)
    else:
        filelist_df = io.load_filelist(sys.stdin)
    
    if args.tagkeys:
        tags_df = io.load_tagkeys(args.tagkeys)
    else:
        tags_df = None

    output_df = main(filelist_df,tags_df=tags_df)

    if args.out:
        outloc = open(args.out,'w')
    else:
        outloc = sys.stdout
    pd.set_option('max_colwidth',int(1e8)) # make sure columns are not truncated
    output_df.to_string(
        outloc, index=False,col_space=10,float_format=utils.format_string)

# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('preprocess')
    p.add_argument(
        '-i', '--i', default=None,help='''Input file, otherwise input
        through the standard input.''')
    p.add_argument('--tagkeys',default=None)
    p.add_argument('-o', '--out', default=None)
    p.set_defaults(func=wrapper)
