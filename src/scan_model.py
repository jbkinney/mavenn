#!/usr/bin/env python

'''A script which accepts an model of binding energy and a wild type sequence.
    The script scans the model across the sequence, and generates an energy
    prediction for each starting position. It then sorts by best binding and 
    displays all posibilities.'''
from __future__ import division
#Our standard Modules
import argparse
import numpy as np
import scipy as sp
import sys
#Our miscellaneous functions
import pandas as pd
import sst.utils as utils
import sst.Models as Models
import sst.io as io
import sst.qc as qc
import re
import pdb
from sst import SortSeqError

def main(model_df, contig_dict):

    # Determine type of string from model
    qc.validate_model(model_df)
    seqtype, modeltype = qc.get_model_type(model_df)
    seq_dict,inv_dict = utils.choose_dict(seqtype,modeltype=modeltype)

    # Check that all characters are from the correct alphabet
    alphabet = qc.seqtype_to_alphabet_dict[seqtype]
    search_string = r"[^%s]"%alphabet
    for contig_name, contig_str in contig_dict.items():
        if re.search(search_string,contig_str):
            raise SortSeqError(\
                'Invalid character for seqtype %s found in %s.'%\
                (seqtype,contig_name))

    # Create model object to evaluate on seqs
    if modeltype == 'MAT':
        model_obj = Models.LinearModel(model_df)
    elif modeltype == 'NBR':
        model_obj = Models.NeighborModel(model_df)
    
    # Create list of dataframes, one for each contig
    seq_col = qc.seqtype_to_seqcolname_dict[seqtype]
    L = model_obj.length
    these_dfs = []
    for contig_name, contig_str in contig_dict.items():
        if len(contig_str) < L:
            continue
        this_df = pd.DataFrame(\
            columns=['val',seq_col,'left','right','ori','contig'])
        num_sites = len(contig_str) - L + 1
        lefts = np.arange(num_sites).astype(int)
        this_df['left'] = lefts
        this_df['right']  = lefts + L - 1
        this_df[seq_col] = [contig_str[i:(i+L)] for i in lefts]
        this_df['ori'] = '+'
        this_df['contig'] = contig_name
        this_df['val'] = model_obj.evaluate(this_df[seq_col])
        these_dfs.append(this_df.copy())

        # If scanning DNA, scan reverse-complement as well
        if seqtype=='dna':
            this_df[seq_col] = [qc.rc(s) for s in this_df[seq_col]]
            this_df['ori'] = '-'
            this_df['val'] = model_obj.evaluate(this_df[seq_col])
            these_dfs.append(this_df.copy())

    # If no sites were found, raise error
    if len(these_dfs)==0:
        raise SortSeqError(\
            'No full-length sites found within provided contigs.')

    # Concatenate dataframes
    sitelist_df = pd.concat(these_dfs,ignore_index=True)

    # Sort by value and reindex
    sitelist_df.sort(columns='val', ascending=False, inplace=True)

    sitelist_df = qc.validate_sitelist(sitelist_df,fix=True)
    return sitelist_df


def wrapper(args):
    """ Wrapper for function for scan_model.main()
    """  
    # Prepare input to main
    model_df = io.load_model(args.model)

    # Create contig_dict from sequences
    if args.seq:
        contig_dict = {'manual':args.seq}
    else:
        raise SortSeqError('No input sequences provided.')

    # Compute results
    outloc = io.validate_file_for_writing(args.out) if args.out else sys.stdout
    output_df = main(model_df,contig_dict)

    # Write df to stdout or to outfile 
    io.write(output_df,outloc,fast=args.fast)

# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('scan_model')
    p.add_argument(
        '-s','--seq',type=str,default=None,
        help ='sequence to scan')
    p.add_argument(
        '-m','--model', help='model to scan sequence with')
    p.add_argument('-o', '--out', default=None)
    p.add_argument(
        '-f','--fast', action='store_true', 
        help="Output is a little harder to read, but is written much faster."
        )
    p.set_defaults(func=wrapper)
