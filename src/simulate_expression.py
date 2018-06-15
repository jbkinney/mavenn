#!/usr/bin/env python

'''Simulates expression from an mpra, selex, or protein selection experiment''' 
from __future__ import division
import argparse
import numpy as np
import scipy as sp
import pandas as pd
import sys
import Models as Models
import utils as utils
import io_local as io
import qc as qc
#from . import SortSeqError
from __init__ import SortSeqError
import evaluate_model as evaluate_model

def main(df,model_df,T_LibCounts,T_mRNACounts,start=0,end=None,mult=1):
   output_df = pd.DataFrame()
   for chunk in df:
       chunk.reset_index(inplace=True,drop=True)
       chunk = evaluate_model.main(chunk,model_df,left=start,right=None)
       #We assume only noise is binomial noise(which we approx as poisson)
       mymodel = Models.PoissonNoise()
       #calculate new expression levels based on energies of each sequence.	 
       libcounts,expcounts = mymodel.gennoisyexp(chunk,T_LibCounts,T_mRNACounts \
       ,mult=mult)
       temp_df = pd.DataFrame()
       lc = pd.Series(libcounts,name='ct_0')
       ec = pd.Series(expcounts,name='ct_1')
       chunk['ct_0'] = lc
       chunk['ct_1'] = ec
       chunk['ct'] = chunk[['ct_0','ct_1']].sum(axis=1)
       chunk = chunk[chunk['ct']!=0]
       del chunk['val']
       output_df = pd.concat([output_df,chunk],axis=0).copy()
   return output_df
    
# Define commandline wrapper
def wrapper(args):
    T_LibCounts = args.totallibcounts
    T_mRNACounts = args.totalmRNAcounts
    if T_LibCounts <=0 or T_mRNACounts <= 0:
        raise SortSeqError('Counts must be greater than zero')
    model_df = io.load_model(args.model)
    if args.i:
        df = pd.io.parsers.read_csv(args.i,delim_whitespace=True \
            ,chunksize=args.chunksize)
    else:
        df = pd.io.parsers.read_csv(sys.stdin,delim_whitespace=True, \
            chunksize=args.chunksize)
   
    output_df = main(df,model_df,T_LibCounts,T_mRNACounts,start=args.start, \
        end=args.end,mult=args.mult)
    
    if args.out:
        outloc = open(args.out,'w')
    else:
        outloc = sys.stdout
    pd.set_option('max_colwidth',int(1e8))

    # Validate dataframe for writting
    output_df.reset_index(inplace=True,drop=True)
    df = qc.validate_dataset(output_df,fix=True)
    io.write(output_df,outloc)
        

# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('simulate_expression')
    p.add_argument(
        '-lC', '--totallibcounts', type=int,default=1000000,help ='''
        Number of Sequencing Counts from your initial library''')
    p.add_argument(
        '-mC', '--totalmRNAcounts', type=int,default=1000000,
        help='''Number of mRNA sequences.''')
    p.add_argument('-m', '--model', default=None,
        help='''
        MAT=FileName,NBR=Filename.
        ''')
    p.add_argument('-cs','--chunksize',default=50000,type=int)
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
    p.add_argument('--mult',type=float,default=1,help='''Number to multiply your
        model output by''')
    p.set_defaults(func=wrapper)
