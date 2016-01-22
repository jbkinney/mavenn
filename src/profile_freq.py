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

def main(dataset_df, bin=None, start=0, end=None):
    """
    Computes character frequencies (0.0 to 1.0) at each position

    Arguments:
        dataset_df (pd.DataFrame): A dataframe containing a valid dataset.
        bin (int): A bin number specifying which counts to use
        start (int): An integer specifying the sequence start position
        end (int): An integer specifying the sequence end position

    Returns:
        freq_df (pd.DataFrame): A dataframe containing counts for each nucleotide/amino acid character at each position. 
    """

    # Validate dataset_df
    qc.validate_dataset(dataset_df)

    # Compute counts
    counts_df = profile_ct.main(dataset_df, bin=bin, start=start, end=end)

    # Create columns for profile_freqs table
    ct_cols = [c for c in counts_df.columns if qc.is_col_type(c,'ct_')]
    freq_cols = ['freq_'+c.split('_')[1] for c in ct_cols]

    # Compute frequencies from counts
    freq_df = counts_df[ct_cols].div(counts_df['ct'], axis=0)
    freq_df.columns = freq_cols
    freq_df['pos'] = counts_df['pos']

    # Validate as counts dataframe
    freq_df = qc.validate_profile_freq(freq_df,fix=True)
    return freq_df


# Define commandline wrapper
def wrapper(args):
    
    # Load dataset dataframe
    if args.i:
        df = io.load_dataset(args.i)
    else:
        df = io.load_dataset(sys.stdin)
    
    # Compute freqs profile dataframe
    output_df = main(df,bin=args.bin,start=args.start,end=args.end)

    # Set output buffer
    if args.out:
        outloc = open(args.out,'w')
    else:
        outloc = sys.stdout

    # Write results to file/stdout
    io.write(out_df,outloc)


# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('profile_freq')
    p.add_argument(
        '-b','--bin',default=None, help='''Dataset bin to use for counts. If blank, total counts will be used''')
    p.add_argument(
        '-i', '--i', default=None,help='''Input file, otherwise input through the standard input.''')
    p.add_argument(
        '-s','--start',type=int,default=0,help ='''Position to start your analyzed region''')
    p.add_argument(
        '-e','--end',type=int,default = None, help='''Position to end your analyzed region''')
    p.add_argument(\
        '-o', '--out', default=None,help='''Output file, otherwise use standard output.''')
    p.set_defaults(func=wrapper)
