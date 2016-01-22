#!/usr/bin/env python
'''
Calculates the fractional number of character occurances at each position within the set of sequences passed.
'''
from __future__ import division
import argparse
import numpy as np
import sys
import pandas as pd
import sst.qc as qc
import sst.io as io
import sst.profile_ct as profile_ct
import pdb

def main(dataset_df, bin=None, start=0, end=None, err=False):
    """
    Computes the mutation rate (0.0 to 1.0) at each position. Mutation rate is defined as 1.0 minus the maximum character frequency at a position. Errors are estimated using bionomial uncertainty

    Arguments:
        dataset_df (pd.DataFrame): A dataframe containing a valid dataset.
        bin (int): A bin number specifying which counts to use
        start (int): An integer specifying the sequence start position
        end (int): An integer specifying the sequence end position

    Returns:
        freq_df (pd.DataFrame): A dataframe containing results. 
    """

    # Validate dataset_df
    qc.validate_dataset(dataset_df)

    # Compute counts
    counts_df = profile_ct.main(dataset_df, bin=bin, start=start, end=end)

    # Create columns for profile_freqs table
    ct_cols = [c for c in counts_df.columns if qc.is_col_type(c,'ct_')]

    # Record positions in new dataframe
    mut_df = counts_df[['pos']].copy()

    # Compute mutation rate across counts
    max_ct = counts_df[ct_cols].max(axis=1)
    sum_ct = counts_df[ct_cols].sum(axis=1)
    mut = 1.0 - (max_ct/sum_ct)
    mut_df['mut'] = mut

    # Computation of error rate is optional
    if err:
        mut_err = np.sqrt(mut*(1.0-mut)/sum_ct)
        mut_df['mut_err'] = mut_err

    # Figure out which alphabet the cts dataframe specifies
    alphabet = ''.join([c.split('_')[1] for c in ct_cols])
    wt_col = None
    for k in qc.seq_alphabets_dict.keys():
        if ('wt' in k) and (alphabet == qc.seq_alphabets_dict[k]):
            wt_col = k
    if wt_col is None:
        raise TypeError('Could not determine wt column name.')

    # Compute WT base at each position
    mut_df[wt_col] = 'X'
    for col in ct_cols:
        indices = (counts_df[col]==max_ct).values
        mut_df.loc[indices,wt_col] = col.split('_')[1]

    # Validate as counts dataframe
    mut_df = qc.validate_profile_mut(mut_df,fix=True)
    return mut_df


# Define commandline wrapper
def wrapper(args):
    
    # Load dataset dataframe
    if args.i:
        df = io.load_dataset(args.i)
    else:
        df = io.load_dataset(sys.stdin)
    
    # Compute mut profile dataframe
    output_df = main(df,bin=args.bin,start=args.start,end=args.end,err=args.d)

    # Set output buffer
    if args.out:
        outloc = open(args.out,'w')
    else:
        outloc = sys.stdout

    # Write results to file/stdout
    io.write(out_df,outloc)


# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('profile_mut')
    p.add_argument(
        '-b','--bin',default=None, help='''Dataset bin to use for counts. If blank, total counts will be used''')
    p.add_argument(
        '-i', '--i', default=None,help='''Input file, otherwise input through the standard input.''')
    p.add_argument(
        '-s','--start',type=int,default=0,help ='''Position to start your analyzed region''')
    p.add_argument(
        '-e','--end',type=int,default = None, help='''Position to end your analyzed region''')
    p.add_argument(
        '-d','--err',type=bool,default = False, help='''Whether or not to include error estimates.''')
    p.add_argument(\
        '-o', '--out', default=None,help='''Output file, otherwise use standard output.''')
    p.set_defaults(func=wrapper)
