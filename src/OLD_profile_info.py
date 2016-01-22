#!/usr/bin/env python

'''A script which calculates the mutual information between base identity
    and batch.'''
from __future__ import division
#Our standard Modules
import argparse
import numpy as np
import sys
import csv
import sst.nsbestimator as nsb
#Our miscellaneous functions
#This module will allow us to easily tally the letter counts at a particular position
import pandas as pd
import sst.utils as utils



def main(df,start,end,dicttype):
    
    seq_dict,inv_dict = utils.choose_dict(dicttype)
    #Check that we have the right dictionary
    def check_sequences(s):
        return set(s).issubset(seq_dict)
    if False in set(df.seq.apply(check_sequences)):
        raise TypeError('Wrong sequence type!')
    Lseq = len(seq_dict)
    #select target sequence region
    df.loc[:,'seq'] = df.loc[:,'seq'].str.slice(start,end)
    #Drop any sequences with incorrect length
    if not end:
        seqL = len(df['seq'][0]) - start
    else:
        seqL = end-start
    df = df[df.seq.apply(len) == (seqL)]
    #If there were sequences of different lengths, then print error
    if len(set(df.seq.apply(len))) > 1:
         sys.stderr.write('Lenghts of all sequences are not the same!')
    #Find number of bins    
    column_headers = utils.get_column_headers(df)
    nbins = len(column_headers)
    MI = np.zeros(seqL)
    MIstd = np.zeros(seqL)
    #interate from 0 to nbins because batches are indexed starting at 0
    pbatch = np.array([np.sum(df['ct_' + str(z)]) for z in range(0,nbins)])
    ct = df[column_headers].sum(axis=1) #total counts for each seq
    for i in range(seqL):
        #Find entries with each base at this position
        letlist = df.loc[:,'seq'].str.slice(i,i+1)
        positions = [letlist==inv_dict[z] for z in range(Lseq)]
        bincounts = np.zeros([Lseq,nbins])
        #tally the number of each base in each bin
        for z in range(nbins):
            bincounts[:,z] = [
                np.sum(df['ct_' + str(z)][positions[q]]) for q in range(Lseq)]
        baseprob = bincounts.sum(axis = 1)/np.sum(ct)
        partialent = 0
        partialvar = 0
        '''use NSB estimator to calculate entropies and errors in entropies.
            Then use these to calculate the mutual information of base identity'''
	for s in range(Lseq):
	    partialent = partialent + baseprob[s]*nsb.S(bincounts[s,:],
                np.sum(bincounts[s,:]),len(bincounts[s,:]))
            partialvar = partialvar + baseprob[s]*nsb.dS(bincounts[s,:],
                np.sum(bincounts[s,:]),len(bincounts[s,:]))
        MI[i] = nsb.S(pbatch,np.sum(pbatch),len(pbatch)) - partialent
        MIstd[i] = np.sqrt(partialvar)
    output_df = pd.concat([pd.Series(range(start,start + seqL)),
        pd.DataFrame(MI),pd.Series(MIstd)],axis=1)
    output_df.columns = ['pos','info','info_err']  
    return output_df

# Define commandline wrapper
def wrapper(args):        
    # Run funciton
    start = args.start
    end = args.end
    dicttype = args.type

    if args.i:
        df = pd.io.parsers.read_csv(args.i,delim_whitespace=True)
    else:
        df = pd.io.parsers.read_csv(sys.stdin,delim_whitespace=True)

    output_df = main(df,start,end,dicttype)
    if args.out:
        outloc = open(args.out,'w')
    else:
        outloc = sys.stdout
    pd.set_option('max_colwidth',int(1e8))
    output_df.to_string(
        outloc, index=False,col_space=10,float_format=utils.format_string)

# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('profile_info')
    p.add_argument(
        '-s','--start',type=int,default=0,help ='''Position to start 
        your analyzed region''')
    p.add_argument(
        '-i','--i',default=None,help='''Input file, otherwise input
        through the standard input.''')
    p.add_argument(
        '-e','--end',type=int,default = None, help='''Position to end 
        your analyzed region''')
    p.add_argument(
        '-t', '--type', choices=['dna','rna','protein'], default='dna')
    p.add_argument('-o', '--out', default=None)
    p.set_defaults(func=wrapper)
