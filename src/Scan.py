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

def main(wtseq,model,dicttype,modeltype='LinearEmat',is_df=True):
    seq_dict,inv_dict = utils.choose_dict(dicttype,modeltype=modeltype)
    #Check to make sure wtseq only contains allowed bases
    lin_seq_dict,lin_inv_dict = utils.choose_dict(dicttype,modeltype='LinearEmat')
    def check_sequences(s):
        return set(s).issubset(lin_seq_dict)
    if not check_sequences(wtseq):
        raise ValueError(
            'Please use only bases contained in ' + str(lin_seq_dict.keys()))

    if modeltype == 'LinearEmat':
        mymodel = Models.LinearModel(model,dicttype,is_df=is_df)
    elif modeltype == 'Neighbor':
        mymodel = Models.NeighborModel(model,dicttype,is_df=is_df)
    
    #Create a list of sequences to evaluate
    seqs = []
    for i in range(len(wtseq)-len(model.index)):
        seqs.append(wtseq[i:i+len(model.index)])
    seqs_series = pd.DataFrame(seqs,columns={'seq'})
    #evaluate model on each sequence.
    energy = pd.Series(mymodel.genexp(seqs),name='val')
    '''Create starting position and length pandas series, then combine with energy
        into one dataframe'''
    pos = pd.Series(range(len(wtseq)-len(model.index)),name='start')
    length = pd.Series([len(model.index) for i in range(len(seqs_series))],name='length')
    output_df = pd.concat([seqs_series,energy,pos,length],axis=1)
    output_df.sort(columns='val',inplace=True)
    
    return output_df

def wrapper(args):
    model = pd.io.parsers.read_csv(args.model,delim_whitespace=True)
    output_df = main(args.wtseq,model,args.type,modeltype=args.modeltype)

    if args.out:
        outloc = open(args.out,'w')
    else:
        outloc = sys.stdout
    pd.set_option('max_colwidth',int(1e8))
    output_df.to_string(
        outloc, index=False,col_space=10,float_format=utils.format_string)

    


# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('Scan')
    p.add_argument(
        '-w','--wtseq',type=str,default=None,
        help ='Wild Type sequence')
    p.add_argument(
        '-m','--model', help='model to use')
    p.add_argument(
        '-t', '--type', choices=['dna','rna','protein'], default='dna')
    p.add_argument(
        '-mt','--modeltype',choices=['LinearEmat','Neighbor'],
        default='LinearEmat',help='Type of Model')
    p.add_argument('-o', '--out', default=None)
    p.set_defaults(func=wrapper)
