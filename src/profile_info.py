#!/usr/bin/env python

'''A script which calculates the mutual information between base identity
    and batch.'''
from __future__ import division
#Our standard Modules
import argparse
import numpy as np
import sys
import csv
#Our miscellaneous functions
#This module will allow us to easily tally the letter counts at a particular position
import pandas as pd
import sortseq_tools.utils as utils
import sortseq_tools.qc as qc
import sortseq_tools.io as io
import sortseq_tools.profile_ct as profile_ct
import sortseq_tools.info as info
import pdb
from sortseq_tools import SortSeqError

def main(dataset_df, err=False, method='naive',\
    pseudocount=1.0, start=0, end=None):
    """
    Computes the mutual information (in bits), at each position, between the character and the bin number. 

    Arguments:
        dataset_df (pd.DataFrame): A dataframe containing a valid dataset.
        start (int): An integer specifying the sequence start position
        end (int): An integer specifying the sequence end position
        method (str): Which method to use to estimate mutual information

    Returns:
        info_df (pd.DataFrame): A dataframe containing results.
    """

    # Validate dataset_df
    qc.validate_dataset(dataset_df)

    # Get number of bins
    bin_cols = [c for c in dataset_df.columns if qc.is_col_type(c,'ct_')]
    if not len(bin_cols) >= 2:
        raise SortSeqError('Information profile requires at least 2 bins.')
    bins = [int(c.split('_')[1]) for c in bin_cols]
    num_bins = len(bins)

    # Get number of characters
    seq_cols = [c for c in dataset_df.columns if qc.is_col_type(c,'seqs')]
    if not len(seq_cols)==1:
        raise SortSeqError('Must be only one seq column.') 
    seq_col = seq_cols[0]
    seqtype = qc.colname_to_seqtype_dict[seq_col]
    alphabet = qc.seqtype_to_alphabet_dict[seqtype]
    ct_cols = ['ct_'+a for a in alphabet]
    num_chars = len(alphabet)

    # Get sequence length and check start, end numbers
    num_pos = len(dataset_df[seq_col][0])
    if not (0 <= start < num_pos):
        raise SortSeqError('Invalid start==%d, num_pos==%d'%(start,num_pos))
    if end is None:
        end = num_pos
    elif (end > num_pos):
        raise SortSeqError('Invalid end==%d, num_pos==%d'%(end,num_pos))
    elif end <= start:
        raise SortSeqError('Invalid: start==%d >= end==%d'%(start,end))

    # Record positions in new dataframe
    counts_df = profile_ct.main(dataset_df)
    info_df = counts_df.loc[start:(end-1),['pos']].copy() # rows from start:end
    info_df['info'] = 0.0
    if err:
        info_df['info_err'] = 0.0

    # Fill in 3D array of counts
    ct_3d_array = np.zeros([end-start, num_chars, num_bins])
    for i, bin_num in enumerate(bins):

        # Compute counts
        counts_df = profile_ct.main(dataset_df, bin=bin_num)

        # Fill in counts table
        ct_3d_array[:,:,i] = counts_df.loc[start:(end-1),ct_cols].astype(float)

    # Compute mutual information for each position
    for i in range(end-start): # i only from start:end

        # Get 2D counts
        nxy = ct_3d_array[i,:,:]
        assert len(nxy.shape) == 2

        # Compute mutual informaiton
        if err:
            mi, mi_err = info.estimate_mutualinfo(nxy,err=True,\
                method=method,pseudocount=pseudocount)
            info_df.loc[i+start,'info'] = mi
            info_df.loc[i+start,'info_err'] = mi_err
        else:
            mi = info.estimate_mutualinfo(nxy,err=False,\
                method=method,pseudocount=pseudocount)
            info_df.loc[i+start,'info'] = mi

    # Validate info dataframe
    info_df = qc.validate_profile_info(info_df,fix=True)
    return info_df


# Define commandline wrapper
def wrapper(args):
    """ Commandline wrapper for main()
    """ 
    inloc = io.validate_file_for_reading(args.i) if args.i else sys.stdin
    outloc = io.validate_file_for_writing(args.out) if args.out else sys.stdout
    input_df = io.load_dataset(inloc)
    output_df = main(input_df, start=args.start,end=args.end,\
        err=args.err, method=args.method, pseudocount=args.pseudocount)
    io.write(output_df,outloc)


# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('profile_info')
    p.add_argument(
        '-i', '--i', type=str, default=None, help='''Input file, otherwise input through the standard input.''')
    p.add_argument(
        '-s','--start',type=int, default=0,help ='''Position to start your analyzed region''')
    p.add_argument(
        '-e','--end',type=int, default = None, help='''Position to end your analyzed region''')
    p.add_argument(\
        '-o', '--out', type=str, default=None,help='''Output file, otherwise use standard output.''')
    p.add_argument(
        '-d','--err',type=bool ,default = False, help='''Whether or not to include error estimates.''')
    p.add_argument(
        '-m','--method', type=str, default='naive', choices=['naive','nsb'],\
        help='''Whether or not to include error estimates.''')
    p.add_argument(
        '-p','--pseudocount', type=float, default=1.0, help='''Pseudocount used to compute information values.''')
    p.set_defaults(func=wrapper)
