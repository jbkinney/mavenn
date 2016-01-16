#!/usr/bin/env python

'''Simulates expression from an mpra, selex, or protein selection experiment''' 
from __future__ import division
import argparse
import numpy as np
import scipy as sp
import pandas as pd
import sys
import sst.Models as Models
import sst.utils as utils

def main(df,T_LibCounts,T_mRNACounts):
   #We assume only noise is binomial noise(which we approx as poisson)
   mymodel = Models.PoissonNoise()
   #calculate new expression levels based on energies of each sequence.	 
   libcounts,expcounts = mymodel.gennoisyexp(df,T_LibCounts,T_mRNACounts)
   return libcounts,expcounts
    
# Define commandline wrapper
def wrapper(args):
    T_LibCounts = args.totallibcounts
    T_mRNACounts = args.totalmRNAcounts
    if args.i:
        df = pd.io.parsers.read_csv(args.i,delim_whitespace=True)
    else:
        df = pd.io.parsers.read_csv(sys.stdin,delim_whitespace=True)

    header = df.columns
    libcounts,expcounts = main(df,T_LibCounts,T_mRNACounts)
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
        '-i', '--i', default=None,help='''Input file, otherwise input
        through the standard input.''')
    p.add_argument('-o', '--out', default=None)
    p.set_defaults(func=wrapper)
