#!/usr/bin/env python

'''A script which adds a predicted energy column to an input table. This is
    generated based on a energy model the user provides.''' 
from __future__ import division
import argparse
import numpy as np
import scipy as sp
import pandas as pd
import sys
import pdb
import time

import sortseq_tools.Models as Models
import sortseq_tools.utils as utils
import sortseq_tools.qc as qc
import sortseq_tools.io as io
from sortseq_tools import SortSeqError
from sortseq_tools import shutthefuckup


def main(dataset_df,model_df,left=None,right=None):

    # Validate dataframes
    qc.validate_dataset(dataset_df)
    qc.validate_model(model_df)

    # Detect model type based on columns
    seqtype, modeltype = qc.get_model_type(model_df)
    seqcol = qc.seqtype_to_seqcolname_dict[seqtype]

    # Set start and end  based on left or right
    if not ((left is None) or (right is None)):
        raise SortSeqError('Cannot set both left and right at same time.')
    if not (left is None):
        start = left
        end = start + model_df.shape[0] + (1 if modeltype=='NBR' else 0)
    elif not (right is None):
        end = right 
        start = end - model_df.shape[0] - (1 if modeltype=='NBR' else 0)
    else:
        start = model_df['pos'].values[0]
        end = model_df['pos'].values[-1] + (2 if modeltype=='NBR' else 1)
    assert start < end 

    # Validate start and end positions
    seq_length = len(dataset_df[seqcol][0])
    if start < 0:
        raise SortSeqError('Invalid start=%d'%start)
    if end >= seq_length:
        raise SortSeqError('Invalid end=%d for seq_length=%d'%(end,seq_length))

    #select target sequence region
    out_df = dataset_df.copy()
    out_df.loc[:,'seq'] = out_df.loc[:,'seq'].str.slice(start,end)

    #Create model object of correct type
    if modeltype == 'MAT':
        mymodel = Models.LinearModel(model_df)
    elif modeltype == 'NBR':
        mymodel = Models.NeighborModel(model_df)
    else:
        raise SortSeqError('Unrecognized model type %s'%modeltype)
 
    # Compute values
    out_df['val'] = mymodel.evaluate(out_df)

    # Validate dataframe and return
    return qc.validate_dataset(out_df,fix=True)


# Define commandline wrapper
def wrapper(args):
    inloc = io.validate_file_for_reading(args.i) if args.i else sys.stdin
    dataset_df = io.load_dataset(inloc)
    model_df = io.load_model(args.model)
    output_df = main(dataset_df=dataset_df, model_df=model_df,\
        left=args.left, right=args.right)
    outloc = io.validate_file_for_writing(args.out) if args.out else sys.stdout
    io.write(output_df,outloc,fast=args.fast)

# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('evaluate_model')
    p.add_argument(\
        '-m', '--model', default=None,
        help="Name of file containing model dataframe. Required.")
    p.add_argument( \
        '-i','--i',default=False, \
        help="Name of file containing dataset dataframe. Defaults to stdin.")
    p.add_argument(\
        '-o', '--out', default=None, \
        help="Name of file to write output dataframe. Defaults to stdout.")
    p.add_argument(
        '-l','--left',type=int, default=None,
        help ='''Seq position at which to align the left-side of the model. Defaults to position determined by model dataframe.''')
    p.add_argument(
        '-r','--right',type=int, default=None,
        help ='''Seq position at which to align the right-side of the model. Defaults to position determined by model dataframe.''')
    p.add_argument(
        '-f','--fast', action='store_true', 
        help="Output is a little harder to read, but is written much faster."
        )
    p.set_defaults(func=wrapper)
