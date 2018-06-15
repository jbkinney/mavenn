#!/usr/bin/env python
'''
Calculates the fractional number of character occurances at each position within the set of sequences passed.
'''
from __future__ import division
import argparse
import numpy as np
import sys
import pandas as pd
import mpathic.qc as qc
import mpathic.io as io
import mpathic.profile_ct as profile_ct
import mpathic.utils as utils
import mpathic.profile_mut as profile_mut
import mpathic.profile_freq as profile_freq
import pdb
from mpathic import SortSeqError

def main(dataset_df, bin=None, start=0, end=None,bins_df=None,pseudocounts=1,
    return_profile=False):
    """
    Computes character frequencies (0.0 to 1.0) at each position

    Arguments:
        dataset_df (pd.DataFrame): A dataframe containing a valid dataset.
        bin (int): A bin number specifying which counts to use
        start (int): An integer specifying the sequence start position
        end (int): An integer specifying the sequence end position

    Returns:
        freq_df (pd.DataFrame): A dataframe containing counts for each 
        nucleotide/amino acid character at each position. 
    """
    seq_cols = qc.get_cols_from_df(dataset_df,'seqs')
    if not len(seq_cols)==1:
        raise SortSeqError('Dataframe has multiple seq cols: %s'%str(seq_cols))
    dicttype = qc.colname_to_seqtype_dict[seq_cols[0]]

    seq_dict,inv_dict = utils.choose_dict(dicttype)
    # Validate dataset_df
    qc.validate_dataset(dataset_df)
    
    #for each bin we need to find character frequency profile, then sum over all
    #bins to get activity.

    #first make sure we have activities of each bin:
    if not bins_df:
        bins = utils.get_column_headers(dataset_df)
        #in this case no activity was specified so just assume the activity
        #equals bin number
        activity = [float(b.split('_')[-1]) for b in bins]
    else:
        bins = list(bins_df['bins'])
        activity = list(bins_df['activity'])

    #initialize dataframe for total counts in all bins
    output_ct_df = pd.DataFrame()
    #initialize dataframe for running activity calculation
    output_activity_df = pd.DataFrame()
    

    for i,b in enumerate(bins): 
        bin_num = int(b.split('_')[-1])
        # Compute counts
        counts_df = profile_ct.main(dataset_df, bin=bin_num, start=start, end=end)

        # Create columns for profile_freqs table
        ct_cols = utils.get_column_headers(counts_df)
        #add_pseudocounts
        counts_df[ct_cols] = counts_df[ct_cols] + pseudocounts
        
        #add to all previous bin counts
        #print output_activity_df
        if i == 0:
            output_ct_df = counts_df[ct_cols]
            output_activity_df = counts_df[ct_cols]*activity[i]
        else:
            output_ct_df = output_ct_df + counts_df[ct_cols]
            output_activity_df = output_activity_df + counts_df[ct_cols]*activity[i]

        
    #now normalize by each character at each position, this is the activity
    #profile
    
    output_activity_df = output_activity_df[ct_cols].div(output_ct_df[ct_cols])
    
    mut_rate = profile_mut.main(dataset_df,bin=bin)
    freq = profile_freq.main(dataset_df,bin=bin)
    freq_cols = [x for x in freq.columns if 'freq_' in x]
    #now normalize by the wt activity
    wtseq = ''.join(mut_rate['wt'])
    wtarr = utils.seq2mat(wtseq,seq_dict)
    
    wt_activity = np.transpose(wtarr)*(np.array(output_activity_df[ct_cols]))
    
    #sum this to get total
    wt_activity2 = wt_activity.sum(axis=1)
    delta_activity = output_activity_df.subtract(pd.Series(wt_activity2),axis=0)
    if return_profile:
        #first find mutation rate according to formula in SI text
        profile_delta_activity = mut_rate['mut']*np.sum(
            (1-np.transpose(wtarr))*np.array(\
            freq[freq_cols])*np.array(delta_activity),axis=1)
        #format into dataframe
        output_df = pd.DataFrame()
        output_df['pos'] = range(start,start+len(profile_delta_activity.index))
        output_df['mut_activity'] = profile_delta_activity
        return output_df
    else:
        #just add pos column and rename counts columns to activity columns
        output_df = pd.DataFrame(delta_activity)
        output_df.insert(0,'pos',range(start,start+len(delta_activity.index)))
        #reorder columns

        activity_col_dict = {x:'activity_' + x.split('_')[-1] \
            for x in delta_activity.columns if 'ct_' in x}
        output_df = output_df.rename(columns=activity_col_dict)
        return output_df


# Define commandline wrapper
def wrapper(args):
    """ Commandline wrapper for main()
    """ 
    inloc = io.validate_file_for_reading(args.i) if args.i else sys.stdin
    outloc = io.validate_file_for_writing(args.out) if args.out else sys.stdout
    if args.bins_df_name:
        bins_df = pd.io.parsers.read_csv(args.bins_df_name,delim_whitespace=True)
    else:
        bins_df = None
    input_df = io.load_dataset(inloc)
    output_df = main(
        input_df,bin=args.bin,start=args.start,end=args.end,bins_df=bins_df,
        pseudocounts=args.pseudocounts,return_profile=args.return_profile)
    io.write(output_df,outloc)


# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('profile_activity')
    p.add_argument(
        '-pc','--pseudocounts', type=int, default=1, help='''pseudocounts
        to add''')
    p.add_argument(
        '-rp','--return_profile', action='store_true', help='''return
        activity profile''')
    p.add_argument(
        '-bdf','--bins_df_name', type=int, default=None, help='''
        name for activity file, if None then automatically assign activity. ''')
    p.add_argument(
        '-b','--bin', type=int, default=None, help='''
        Dataset bin to use for counts. If blank, total counts will be used''')
    p.add_argument(
        '-i', '--i', type=str, default=None, help='''
        Input file, otherwise input through the standard input.''')
    p.add_argument(
        '-s','--start',type=int, default=0,help ='''
        Position to start your analyzed region''')
    p.add_argument(
        '-e','--end',type=int, default = None, help='''
        Position to end your analyzed region''')
    p.add_argument(\
        '-o', '--out', type=str, default=None,help='''
        Output file, otherwise use standard output.''')
    p.set_defaults(func=wrapper)
    
