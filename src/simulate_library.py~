#!/usr/bin/env python

'''A script with generates Simulated Data for a Sort Seq Experiment 
    with a given mutation rate and wild type sequence.''' 
from __future__ import division
import argparse
import numpy as np
import scipy as sp
import pandas as pd
import sys
import sortseq_tools.utils as utils
import sortseq_tools.qc as qc
import sortseq_tools.io as io
from sortseq_tools import SortSeqError
import pdb
from numpy.random import choice

def seq2arr(seq,seq_dict):
    '''Change base pairs to numbers'''
    return np.array([seq_dict[let] for let in seq])

def arr2seq(arr,inv_dict):
    '''Change numbers back into base pairs.'''
    return ''.join([inv_dict[num] for num in arr])

# main function for simulating library
def main(wtseq=None, mutrate=0.10, numseq=10000,dicttype='dna',probarr=None,
        tags=False,tag_length=10):
    
    #generate sequence dictionary
    seq_dict,inv_dict = utils.choose_dict(dicttype)    
                
    mutrate = float(mutrate)
    if (mutrate < 0.0) or (mutrate > 1.0):
        raise SortSeqError('Invalid mutrate==%f'%mutrate)

    numseq = int(numseq)
    if (numseq <= 0):
        raise SortSeqError('numseq must be positive. Is %d'%numseq)

    tag_length = int(tag_length)
    if (tag_length <= 0):
        raise SortSeqErorr('tag_length must be positive. Is %d'%tag_length)

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
        lin_seq_dict,lin_inv_dict = utils.choose_dict(dicttype,modeltype='MAT')
        def check_sequences(s):
            return set(s).issubset(lin_seq_dict)
        if not check_sequences(wtseq):
            raise SortSeqError(
                'wtseq can only contain bases in ' + str(lin_seq_dict.keys()))        
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

    seq_col = qc.seqtype_to_seqcolname_dict[dicttype]
    seqs_df = pd.DataFrame(seqs, columns=[seq_col])

    # If simulating tags, each generated seq gets a unique tag
    if tags:
        tag_seq_dict,tag_inv_dict = utils.choose_dict('dna')
        tag_alphabet_list = tag_seq_dict.keys()

        # Make sure tag_length is long enough for the number of tags needed
        if len(tag_alphabet_list)**tag_length < 2*numseq:
            raise SortSeqError(\
                'tag_length=%d is too short for num_tags_needed=%d'%\
                (tag_length,numseq))

        # Generate a unique tag for each unique sequence
        tag_set = set([])
        while len(tag_set) < numseq:
            num_tags_left = numseq - len(tag_set)
            new_tags = [''.join(choice(tag_alphabet_list,size=tag_length)) \
                for i in range(num_tags_left)]
            tag_set = tag_set.union(new_tags)

        df = seqs_df.copy()
        df.loc[:,'ct'] = 1
        df.loc[:,'tag'] = list(tag_set)

    # If not simulating tags, list only unique seqs w/ corresponding counts
    else:
        seqs_counts = seqs_df[seq_col].value_counts()
        df = seqs_counts.reset_index()
        df.columns = [seq_col,'ct']

    # Convert into valid dataset dataframe and return
    return qc.validate_dataset(df,fix=True)

# Define commandline wrapper
def wrapper(args):
    """ Commandline wrapper for main()
    """  
    output_df = main(wtseq=args.wtseq, mutrate=args.mutrate,\
        numseq=args.numseqs,dicttype=args.type,tags=args.tags,\
        tag_length=args.tag_length)
    outloc = io.validate_file_for_writing(args.out) if args.out else sys.stdout
    io.write(output_df,outloc)

# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('simulate_library')
    p.add_argument(
        '-w', '--wtseq', type=str,help ='Wild Type Sequence')
    p.add_argument(
        '-m', '--mutrate', type=float, default=0.09,
        help='''Mutation Rate, given fractionally. 
        For example enter .1, not 10 percent''')
    p.add_argument('-n', '--numseqs', type=int, default=100,
        help='Number of Sequences')
    # p.add_argument(
    #     '-bp','--baseprob',default=None,help=''' If you would like to 
    #     use custom base probabilities, this is the filename of a 
    #     probability array.''')
    p.add_argument('-tags','--tags',action='store_true',help='''Simulate Tags''')
    p.add_argument('-tl','--tag_length', default=10, type=int, help='''Length of Tag''')
    p.add_argument(
        '-t', '--type', choices=qc.seqtypes, default='dna')
    p.add_argument('-o', '--out', default=None)
    p.set_defaults(func=wrapper)
    return p
