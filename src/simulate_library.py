#!/usr/bin/env python

'''A script with generates Simulated Data for a Sort Seq Experiment 
    with a given mutation rate and wild type sequence.''' 
from __future__ import division
import argparse
import numpy as np
import scipy as sp
import pandas as pd
import sys
import sortseq.utils as utils

def seq2arr(seq,seq_dict):
    '''Change base pairs to numbers'''
    return np.array([seq_dict[let] for let in seq])

def arr2seq(arr,inv_dict):
    '''Change numbers back into base pairs.'''
    return ''.join([inv_dict[num] for num in arr])

# main function for simulating library
def main(wtseq=None, mutrate=0.10, numseq=10000,dicttype='dna',probarr=None,
        tags=False,tag_length=10):
    numseq = int(numseq)
    #generate sequence dictionary
    seq_dict,inv_dict = utils.choose_dict(dicttype)    
                
    if isinstance(probarr,np.ndarray):
        L = probarr.shape[1]
        #Generate bases according to provided probability matrix
        letarr = np.zeros([numseq,L])
        for z in range(L):
            letarr[:,z] = np.random.choice(
                range(len(seq_dict)),numseq,p=probarr[:,z]) 
    else:
        parr = []
        wtseq = wtseq.upper()
        L = len(wtseq)
        letarr = np.zeros([numseq,L])
        #Check to make sure the wtseq uses the correct bases.
        lin_seq_dict,lin_inv_dict = utils.choose_dict(dicttype,modeltype='LinearEmat')
        def check_sequences(s):
            return set(s).issubset(lin_seq_dict)
        if not check_sequences(wtseq):
            raise ValueError(
                'Please use only bases contained in ' + str(lin_seq_dict.keys()))        
        #find wtseq array 
        wtarr = seq2arr(wtseq,seq_dict)
        mrate = mutrate/(len(seq_dict)-1) #prob of non wildtype
        #Generate sequences by mutating away from wildtype
        '''probabilities away from wildtype (0 = stays the same, a 3 for 
            example means a C becomes an A, a 1 means C-> G)'''
        parr = np.array(
            [1-(len(seq_dict)-1)*mrate] 
            + [mrate for i in range(len(seq_dict)-1)])  
        #Generate random movements from wtseq
        letarr = np.random.choice(
            range(len(seq_dict)),[numseq,len(wtseq)],p=parr) 
        #Find sequences
        letarr = np.mod(letarr + wtarr,len(seq_dict))
    seqs= []
    #Convert Back to letters
    for i in range(numseq):
        seqs.append(arr2seq(letarr[i,:],inv_dict))  
    seqs_df = pd.DataFrame(seqs, columns=['seq'])
    seqs_counts = seqs_df['seq'].value_counts()
    df = seqs_counts.reset_index()
    df.columns = ['seq','ct']
    #If you also want to simulate sequence tags.
    t = []
    if tags:
        for i in range(df.shape[0]):
            t.append(''.join(np.choice(seq_dict,size=tag_length)))
        df['tag'] = t
    return df


# Define commandline wrapper
def wrapper(args):
    if args.baseprob:
        probarr = np.genfromtxt(args.baseprob)
    else:
        probarr = None
    # Run funciton
    seqs_df = main(wtseq=args.wtseq, mutrate=args.mutrate, 
        numseq=args.numseqs,dicttype=args.type,probarr=probarr,tags=args.tags,
        tag_length=args.tag_length)
    
    if args.out:
        outloc = open(args.out,'w')
    else:
        outloc = sys.stdout
    pd.set_option('max_colwidth',int(1e8))
    seqs_df.to_string(
        outloc, index=False, columns=['seq','ct'],col_space=10,
        float_format=utils.format_string)

# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('simulate_library')
    p.add_argument(
        '-w', '--wtseq', type=str,help ='Wild Type Sequence')
    p.add_argument(
        '-m', '--mutrate', type=float, default=0.09,
        help='''Mutation Rate, given fractionally. 
        For example enter .1, not 10 percent''')
    p.add_argument('-n', '--numseqs', type=int, default=100000,
        help='Number of Sequences')
    p.add_argument(
        '-bp','--baseprob',default=None,help=''' If you would like to 
        use custom base probabilities, this is the filename of a 
        probability array.''')
    p.add_argument('-tags','--tags',action='store_true',help='''Simulate Tags''')
    p.add_argument('-tl','--tag_length',help='''Length of Tag''')
    p.add_argument(
        '-t', '--type', choices=['dna','rna','protein'], default='dna')
    p.add_argument('-o', '--out', default=None)
    p.set_defaults(func=wrapper)
    return p
