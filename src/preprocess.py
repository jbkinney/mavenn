'''A scripts which accepts a dataframe of input files for each bin and outputs
    a single data frame containing all the info. Combining of paired end reads,
    and any quality score filtering must occur before this script runs.'''
from __future__ import division
#Our standard Modules
import argparse
import numpy as np
import sys
import pandas as pd
import utils as utils
import io_local as io
import qc as qc
import re
import pdb
#from . import SortSeqError
#from __init__ import SortSeqError

print(' Called from pre-process.py')

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
        raise SortSeqError('No datasets were loaded')

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
            raise SortSeqError('\
                Dataframe does not contain index_col="%s"'%index_col)
        if not 'ct' in df.columns:
            raise SortSeqError('\
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
def main(filelist_df,tags_df=None,indir='./',seq_type=None):
    """ Merges datasets listed in the filelist_df dataframe
    """

    # Validate filelist
    qc.validate_filelist(filelist_df)

    # Read datasets into dictionary indexed by bin number
    dataset_df_dict = {}
    for item in filelist_df.iterrows():
        # Autodetect fasta, fastq, or text file based on file extension
        fn = indir+item[1]['file']
        b = item[1]['bin']
        if re.search(fasta_filename_patterns,fn):
            df = io.load_dataset(fn,file_type='fasta',seq_type=seq_type)
        elif re.search(fastq_filename_patterns,fn):
            df = io.load_dataset(fn,file_type='fastq',seq_type=seq_type)
        else:
            df = io.load_dataset(fn,file_type='text',seq_type=seq_type)
        dataset_df_dict[b] = df

    # Merge datasets into one
    out_df = merge_datasets(dataset_df_dict)

    # Add seqs if given tags_df
    if not tags_df is None:
        qc.validate_tagkey(tags_df)
        tag_col = 'tag'

        # Test to make sure all tags in dataset are a subset of tags
        data_tags = set(out_df[tag_col])
        all_tags = set(tags_df[tag_col])
        if not (data_tags <= all_tags):
            sys.stderr.write('Some tags probably could not be identified.')

        # Get name of seq column       
        seq_cols = qc.get_cols_from_df(tags_df, 'seqs')
        if not len(seq_cols)==1:
            raise SortSeqError('Multiple seq columns; exaclty 1 required.')
        seq_col = seq_cols[0]

        # Set tag to be index column of dataframe
        tags_df = tags_df.set_index(tag_col)

        # Add seqs corresponding to each tag
        tags = out_df[tag_col]
        seqs = tags_df[seq_col][tags].values
        if not all([type(x)==str for x in seqs]):
            raise SortSeqError('Some looked-up seqs are not strings.')
        out_df[seq_col] = tags_df[seq_col][tags].values

    qc.validate_dataset(out_df)
    return out_df


# Define commandline wrapper
def wrapper(args):
    """ Commandline wrapper for main()
    """  

    inloc = io.validate_file_for_reading(args.i) if args.i else sys.stdin
    outloc = io.validate_file_for_writing(args.out) if args.out else sys.stdout
    
    # Get filelist
    filelist_df = io.load_filelist(inloc)
    inloc.close()

    # Get tagkeys dataframe if provided
    if args.tagkeys:
        tagloc = io.validate_file_for_reading(args.tagkeys) 
        tags_df = io.load_tagkey(tagloc)
        tagloc.close()
    else:
        tags_df = None
    
    output_df = main(filelist_df,tags_df=tags_df,seq_type=args.seqtype)
    io.write(output_df,outloc,fast=args.fast)


# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('preprocess')
    p.add_argument(
        '-i', '--i', type=str, default=None, help='''Input file, otherwise input through the standard input.''')
    p.add_argument('--tagkeys',default=None,help='''If there is a tag-key file,
        supply the file name after this flag''')
    p.add_argument(\
        '-o', '--out', type=str, default=None,help='''Output file, otherwise use standard output.''')
    p.add_argument(
        '-s', '--seqtype', default=None, choices=['dna','rna','protein'], \
        help='''Type of sequence to expect in input files.''')
    p.add_argument(
        '-f','--fast', action='store_true', 
        help="Output is a little harder to read, but is written much faster."
        )
    p.set_defaults(func=wrapper)
