#!/usr/bin/env python

'''A script which adds a predicted energy column to an input table. This is
    generated based on a energy model the user provides.''' 
from __future__ import division
import argparse
import numpy as np
import scipy as sp
import pandas as pd
import sys

import sst.Models as Models
import sst.utils as utils

def main(df,mp,dicttype,modeltype='MAT',is_df=False,start=0,end=None):
    #Create sequence dictionary
    seq_dict,inv_dict = utils.choose_dict('dna')
    #Check to make sure the chosen dictionary type correctly describes the sequences
    lin_seq_dict,lin_inv_dict = utils.choose_dict(dicttype,modeltype='MAT')
    def check_sequences(s):
        return set(s).issubset(lin_seq_dict)
    if False in set(df.seq.apply(check_sequences)):
        raise TypeError('Wrong sequence type!')
    #select target sequence region
    df.loc[:,'seq'] = df.loc[:,'seq'].str.slice(start,end)
    #Create model object of correct type
    if modeltype == 'MAT':
        mymodel = Models.LinearModel(mp,dicttype,is_df=is_df)
    elif modeltype == 'NBR':
        mymodel = Models.NeighborModel(mp,dicttype,is_df=is_df)
    elif modeltype == 'RandomLinear':
        emat_0 = utils.RandEmat(len(df['seq'][0]),len(seq_dict))
        mymodel = Models.LinearModel(emat_0,dicttype)
    #Evaluate Model on sequences    
    df['val'] = mymodel.genexp(df['seq'])
    return df['val']


# Define commandline wrapper
def wrapper(args):
    modeltype = args.modeltype
    dicttype = args.type
    # Run funciton
    if args.i:
        df = pd.io.parsers.read_csv(args.i,delim_whitespace=True)
    else:
        df = pd.io.parsers.read_csv(sys.stdin,delim_whitespace=True)
    print len(df['seq'][0])
    df['val'] = main(df,args.mp,dicttype,modeltype=modeltype,is_df=args.DataFrame)
    
    
    

    if args.out:
        outloc = open(args.out,'w')
    else:
        outloc = sys.stdout
    pd.set_option('max_colwidth',int(1e8))
    df.to_string(
        outloc, index=False,col_space=10,float_format=utils.format_string)

# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('simulate_evaluate')
    p.add_argument(
        '-m', '--modeltype', type=str,choices=['RandomLinear','MAT',
        'NBR'],help ='Type of Model to use')
    p.add_argument('-mp', '--modelparam', default=None,
        help=''' 
        RandomLinear=LengthofSeq,MAT=FileName, NBR=FileName. ''')
    p.add_argument(
        '-t', '--type', choices=['dna','rna','protein'], default='dna')
    p.add_argument(
        '-i','--i',default=False,help='''Read input from file instead 
        of stdin''')
    p.add_argument('-df','--DataFrame', action='store_true')
    p.add_argument('-o', '--out', default=None)
    p.add_argument(
        '-s','--start',type=int,default=0,
        help ='Position to start your analyzed region')
    p.add_argument(
        '-e','--end',type=int,default = None,
        help='Position to end your analyzed region')
    p.set_defaults(func=wrapper)
