#!/usr/bin/env python

'''A script which returns the mutual information between the predictions of a
    model and a test data set.'''

from __future__ import division
#Our standard Modules
import argparse
import numpy as np
import scipy as sp
import sys
import pandas as pd
#Our miscellaneous functions
#This module will allow us to easily tally the letter counts at a particular position

import sst.utils as utils
import sst.EstimateMutualInfoforMImax as EstimateMutualInfoforMImax


def main(
        data_df,model_df,dicttype='dna',exptype=None,modeltype='LinearEmat',
        start=0,end=None,no_err=False):
    seq_dict,inv_dict = utils.choose_dict(dicttype,modeltype)
    if (start != 0 or end):
        data_df.loc[:,'seq'] = data_df.loc[:,'seq'].str.slice(start,end)
    col_headers = utils.get_column_headers(data_df)
    if 'ct' not in data_df.columns:
                data_df['ct'] = data_df[col_headers].sum(axis=1)
    data_df = data_df[data_df.ct != 0]        
    if not end:
        seqL = len(data_df['seq'][0]) - start
    else:
        seqL = end-start
    data_df = data_df[data_df.seq.apply(len) == (seqL)] 
    #make a numpy array out of the model data frame
    model_df_headers = ['val_' + str(inv_dict[i]) for i in range(len(seq_dict))]
    value = np.transpose(np.array(model_df[model_df_headers]))  
    #now we evaluate the expression of each sequence according to the model.
    dot = np.zeros(len(data_df.index))
    if modeltype == 'LinearEmat':
        for i,s in enumerate(data_df['seq']):
            dot[i] = np.sum(value*utils.seq2mat(s,seq_dict))
        data_df['val'] = dot                   
    elif modeltype=='Neighbor':
        for i,s in enumerate(data_df['seq']):
            dot[i] = np.sum(value*utils.seq2matpair(s,seq_dict))
        data_df['val'] = dot
    else:
        raise ValueError('Cannot handle other model types at the moment. Sorry!')
    df_sorted = data_df.sort(columns='val')
    df_sorted.reset_index(inplace=True)
    #we must divide by the total number of counts in each bin for the MI calculator
    df_sorted[col_headers] = df_sorted[col_headers].div(df_sorted['ct'],axis=0)     
    MI = EstimateMutualInfoforMImax.alt2(df_sorted)
    if no_err:
        Std = np.NaN
    else:
        data_df_for_sub = data_df.copy()
        sub_MI = np.zeros(15)
        for i in range(15):
            sub_df = data_df_for_sub.sample(int(len(data_df_for_sub.index)/2))
            sub_df.reset_index(inplace=True)
            sub_MI[i],sub_std = main(
                sub_df,model_df,dicttype=dicttype,modeltype=modeltype,no_err=True)
        Std = np.std(sub_MI)/np.sqrt(2)
    return MI,Std
     
def wrapper(args):
    
    data_df = pd.io.parsers.read_csv(args.dataset,delim_whitespace=True)    	    
    # Take input from standard input or through the -i flag.
    if args.model:
        model_df = pd.io.parsers.read_csv(args.model,delim_whitespace=True)
    else:
        model_df = pd.io.parsers.read_csv(sys.stdin,delim_whitespace=True)
    MI,Std = main(
        data_df,model_df,dicttype=args.type,exptype=args.exptype,start=args.start,
        end=args.end,modeltype=args.modeltype,no_err=args.no_err)
    output_df = pd.DataFrame([MI],columns=['info'])
    output_df = pd.concat([output_df,pd.Series(Std,name='err')],axis=1)
  
    if args.out:
        outloc = open(args.out,'w')
    else:
        outloc = sys.stdout
    pd.set_option('max_colwidth',int(1e8))
    output_df.to_string(
        outloc, index=False,col_space=10,float_format=utils.format_string)

# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('predictiveinfo')
    p.add_argument('-ds','--dataset')
    p.add_argument(
        '-mt','--modeltype',default='LinearEmat',
        choices=['LinearEmat','Neighbor'],help='''Type of model to be evaluated''')
    p.add_argument(
        '-expt','--exptype',default=None,choices=[None,'sortseq','selex',
        'dms','mpra'])
    p.add_argument(
        '-t', '--type', choices=['dna','rna','protein'], default='dna')
    p.add_argument(
        '--no_err',action='store_true',help='''Flag to use if you do not want to
        calculate error''')
    p.add_argument(
        '-s','--start',type=int,default=0,help ='''Position to start your 
        analyzed region''')
    p.add_argument(
        '-e','--end',type=int,default = None, 
        help='''Position to end your analyzed region''')
    p.add_argument(
        '-m', '--model', default=None,help='''Model file, otherwise input
        through the standard input.''')            
    p.add_argument('-o', '--out', default=None)
    p.set_defaults(func=wrapper)
