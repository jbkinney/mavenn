#!/usr/bin/env python

'''A script which accepts a library through standard input and outputs 
    mutation rate across each position.'''
from __future__ import division
#Our standard Modules
import argparse
import numpy as np
import sys
import pandas as pd
import sst.utils as utils



def main(df,dicttype):
    #create sequence dictionary, do not select a paired version
    seq_dict,inv_dict = utils.choose_dict(dicttype)
    #Check to make sure the chosen dictionary type correctly describes the sequences
    lin_seq_dict,lin_inv_dict = utils.choose_dict(dicttype,modeltype='LinearEmat')
    def check_sequences(s):
        return set(s).issubset(lin_seq_dict)
    if False in set(df.seq.apply(check_sequences)):
        raise TypeError('Wrong sequence type!')
    #column_headers which describe all counts columns in the dataframe
    column_headers = ['ct_' + inv_dict[i] for i in range(len(seq_dict))]
    L = len(df['pos'])
    #convert to numpy array
    counts_arr = np.array(df[column_headers])
    total_counts = counts_arr.sum()
    T_counts = np.sum(counts_arr,axis=1)[0]
    wtarr = np.argmax(counts_arr,axis=1)
    mutprofile = np.zeros(L)
    muterr = np.zeros(L)
    for z in range(L):
        mutprofile[z] = (T_counts-counts_arr[z,wtarr[z]])/T_counts
        #use normal approximation to binomial uncertainty
        muterr[z] = .975*np.sqrt(1/total_counts*mutprofile[z]*(1-mutprofile[z]))
    output_df = pd.DataFrame()
    output_df['pos'] = df['pos']
    output_df['mut'] = mutprofile
    output_df['mut_err'] = muterr
    return output_df

# Define commandline wrapper
def wrapper(args):      
    # Run funciton
    if args.i:
        df = pd.io.parsers.read_csv(args.i,delim_whitespace=True)
    else:
        df = pd.io.parsers.read_csv(sys.stdin,delim_whitespace=True)
    
    output_df = main(df,args.type)
    if args.out:
        outloc = open(args.out,'w')
    else:
        outloc = sys.stdout
    pd.set_option('max_colwidth',int(1e8))
    output_df.to_string(
        outloc, index=False,col_space=10,float_format=utils.format_string)

# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('profile_mutrate')
    p.add_argument(
        '-t', '--type', choices=['dna','rna','protein'], default='dna')
    p.add_argument(
        '-i', '--i', default=None,help='''Input file, otherwise input
        through the standard input.''')
    p.add_argument('-o', '--out', default=None)
    p.set_defaults(func=wrapper)
