#!/usr/bin/env python

'''Simulate cell sorting based on expression'''
from __future__ import division
import argparse
import numpy as np
import scipy as sp
import pandas as pd
import sys
import sortseq_tools.Models as Models
import sortseq_tools.utils as utils
import sortseq_tools.io as io
import sortseq_tools.evaluate_model as evaluate_model
from sortseq_tools import SortSeqError

def main(
    df,mp,noisetype,npar,nbins,sequence_library=True,
    start=0,end=None):
    #validate noise parameters
    if not isinstance(npar,list):
        raise SortSeqError('Noise parameters must be given as a list')
    if noisetype == 'Normal':
        if len(npar) != 1:
            raise SortSeqError('''For a normal noise model, there must be one 
                 input parameter (width of normal distribution)''')
    if noisetype == 'LogNormal':
        if len(npar) != 2:
             raise SortSeqError('''For a LogNormal noise model there must 
                 be 2 input parameters''')
    if nbins <= 1:
        raise SortSeqError('number of bins must be greater than 1')
    #generate predicted energy of each sequence.
    df = evaluate_model.main(df,mp,left=start,right=None)
    #Determine model type to use for noise
    if noisetype == 'LogNormal':
        NoiseModelSort = Models.LogNormalNoise(npar)
    elif noisetype == 'Normal':
        NoiseModelSort = Models.NormalNoise(npar)
    elif noisetype == 'None':
        NoiseModelSort = Models.NormalNoise([1e-16])
    else:
        NoiseModelSort = Models.CustomModel(noisetype,npar)
    #Apply noise to our calculated energies
    noisyexp,listnoisyexp = NoiseModelSort.genlist(df)
    #Determine Expression Cutoffs for bins
    noisyexp.sort()
    cutoffs = list(
        noisyexp[np.linspace(0,len(noisyexp),nbins,endpoint=False,dtype=int)])
    cutoffs.append(np.inf)
    seqs_arr = np.zeros([len(listnoisyexp),nbins],dtype=int)
    #split sequence into bins based on calculated cutoffs
    for i,entry in enumerate(listnoisyexp):
        seqs_arr[i,:] = np.histogram(entry,bins=cutoffs)[0]
    col_labels = ['ct_' + str(i+1) for i in range(nbins)]
    if sequence_library:
        df['ct_0'] = utils.sample(df['ct'],int(df['ct'].sum()/nbins))
    output_df = pd.concat([df,pd.DataFrame(seqs_arr,columns=col_labels)],axis=1)      
    
    return output_df


# Define commandline wrapper
def wrapper(args):
    
    try:
        npar = args.noiseparam.strip('[').strip(']').split(',')
    except:
        npar = []
    nbins = args.nbins
    # Run funciton
    if args.i:
        df = pd.io.parsers.read_csv(
            args.i,delim_whitespace=True,
            dtype={'seqs':str,'batch':int})
    else:
        df = pd.io.parsers.read_csv(
            sys.stdin,delim_whitespace=True,
            dtype={'seqs':str,'batch':int})
    if len(utils.get_column_headers(df)) > 0:
         raise SortSeqError('Library already sorted!')
    model_df = io.load_model(args.model)
    output_df = main(
        df,model_df,args.noisemodel,npar,
        nbins,start=args.start,end=args.end)
    
    if args.out:
        outloc = open(args.out,'w')
    else:
        outloc = sys.stdout
    pd.set_option('max_colwidth',int(1e8))
    output_df.to_string(
        outloc, index=False,col_space=10,float_format=utils.format_string)

# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('simulate_sort')
    p.add_argument('-nm', '--noisemodel',
        choices=['LogNormal','Normal','None'],default='Normal',
        help='''Noise Model to use.''')
    p.add_argument(
        '-npar','--noiseparam',default = '[.2]',help = '''
        Parameters for your noise model, as a list. The required parameters are
        LogNormal=[autoflouro,scale],Normal=[scale].''')
    p.add_argument(
        '-mt', '--modeltype', type=str,choices=['RandomLinear','MAT'
        ,'NBR'],default='MAT',help ='Type of Model to use')
    p.add_argument('-m', '--model', default=None,
        help='''
        MAT=FileName,NBR=Filename.
        ''')
    p.add_argument(
        '-i','--i',default=False,help='''Read input from file instead 
        of stdin''')
    p.add_argument(
        '-n','--nbins',type=int,default=3,
        help='''Number of bins to sort into.''')
    p.add_argument(
         '-sl','--sequence_library',action='store_true',help='''If you
         would also like to simulate sequencing the library in bin zero, select
         this option''')
    p.add_argument(
        '-t', '--type', choices=['dna','rna','protein'], default='dna')
    p.add_argument(
        '-s','--start',type=int,default=0,
        help ='Position to start your analyzed region')
    p.add_argument(
        '-e','--end',type=int,default = None,
        help='Position to end your analyzed region')
    p.add_argument('-o', '--out', default=None)
    p.set_defaults(func=wrapper)
