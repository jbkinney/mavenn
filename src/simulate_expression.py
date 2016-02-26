#!/usr/bin/env python

'''Simulates expression from an mpra, selex, or protein selection experiment''' 
from __future__ import division
import argparse
import numpy as np
import scipy as sp
import pandas as pd
import sys
import sortseq_tools.Models as Models
import sortseq_tools.utils as utils
import sortseq_tools.io as io
from sortseq_tools import SortSeqError
import sortseq_tools.evaluate_model as evaluate_model

def main(df,model_df,T_LibCounts,T_mRNACounts,start=0,end=None):
   df = evaluate_model.main(df,model_df,left=start,right=None)
   #We assume only noise is binomial noise(which we approx as poisson)
   mymodel = Models.PoissonNoise()
   #calculate new expression levels based on energies of each sequence.	 
   libcounts,expcounts = mymodel.gennoisyexp(df,T_LibCounts,T_mRNACounts)
   return libcounts,expcounts
    
# Define commandline wrapper
def wrapper(args):
    T_LibCounts = args.totallibcounts
    T_mRNACounts = args.totalmRNAcounts
    if T_LibCounts <=0 or T_mRNACounts <= 0:
        raise SortSeqError('Counts must be greater than zero')
    model_df = io.load_model(args.model)
    if args.i:
        df = pd.io.parsers.read_csv(args.i,delim_whitespace=True)
    else:
        df = pd.io.parsers.read_csv(sys.stdin,delim_whitespace=True)
    #make sure the library is not already sorted
    if len(utils.get_column_headers(df)) > 0:
         raise SortSeqError('Library already sorted!')
    header = df.columns
    libcounts,expcounts = main(df,model_df,T_LibCounts,T_mRNACounts,start=args.start,end=args.end)
    #add these counts to input dataframe
    lc = pd.Series(libcounts,name='ct_0')
    ec = pd.Series(expcounts,name='ct_1')
    df['ct_0'] = lc
    df['ct_1'] = ec
    if args.out:
        outloc = open(args.out,'w')
    else:
        outloc = sys.stdout
    pd.set_option('max_colwidth',int(1e8))
    df.to_string(
        outloc, index=False,col_space=10,float_format=utils.format_string)

# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('simulate_expression')
    p.add_argument(
        '-lC', '--totallibcounts', type=int,default=1000000,help ='''
        Number of Sequencing Counts from your initial library''')
    p.add_argument(
        '-mC', '--totalmRNAcounts', type=int,default=1000000,
        help='''Number of mRNA sequences.''')
    p.add_argument(
        '-mt', '--modeltype', type=str,choices=['RandomLinear','MAT'
        ,'NBR'],default='MAT',help ='Type of Model to use')
    p.add_argument('-m', '--model', default=None,
        help='''
        MAT=FileName,NBR=Filename.
        ''')
    p.add_argument(
        '-s','--start',type=int,default=0,
        help ='Position to start your analyzed region')
    p.add_argument(
        '-e','--end',type=int,default = None,
        help='Position to end your analyzed region')
    p.add_argument(
        '-i', '--i', default=None,help='''Input file, otherwise input
        through the standard input.''')
    p.add_argument('-o', '--out', default=None)
    p.set_defaults(func=wrapper)
