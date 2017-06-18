#!/usr/bin/env python

'''Simulate cell sorting based on expression'''
from __future__ import division
import argparse
import numpy as np
import scipy as sp
import pandas as pd
import sys
import mpathic.Models as Models
import mpathic.utils as utils
import mpathic.io as io
import mpathic.qc as qc
import mpathic.evaluate_model as evaluate_model
from mpathic import SortSeqError

def main(
    df,mp,noisetype,npar,nbins,sequence_library=True,
    start=0,end=None,chunksize=50000):
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
    

    #determine cutoffs for bins now
    #do progressive sum to try to find cutoffs so their will be equal numbers in each bin
    
        
    #Determine model type to use for noise
    if noisetype == 'LogNormal':
        NoiseModelSort = Models.LogNormalNoise(npar)
    elif noisetype == 'Normal':
        NoiseModelSort = Models.NormalNoise(npar)
    elif noisetype == 'None':
        NoiseModelSort = Models.NormalNoise([1e-16])
    elif noisetype == 'Plasmid':
        NoiseModelSort = Models.PlasmidNoise()
    else:
        NoiseModelSort = Models.CustomModel(noisetype,npar)
    i = 0
    output_df = pd.DataFrame()
    for chunk in df:
        print i
        chunk.reset_index(inplace=True,drop=True)
        chunk = evaluate_model.main(chunk,mp,left=start,right=None)
        
        

        
    
        #Apply noise to our calculated energies
        noisyexp,listnoisyexp = NoiseModelSort.genlist(chunk)
        if i==0:
            #Determine Expression Cutoffs for bins
            noisyexp.sort()
            val_cutoffs = list(
                noisyexp[np.linspace(0,len(noisyexp),nbins,endpoint=False,dtype=int)])
            val_cutoffs.append(np.inf)
            val_cutoffs[0] = -np.inf
        print val_cutoffs
        #Determine Expression Cutoffs for bins
        seqs_arr = np.zeros([len(listnoisyexp),nbins],dtype=int)
        #split sequence into bins based on calculated cutoffs
        for i,entry in enumerate(listnoisyexp):
            seqs_arr[i,:] = np.histogram(entry,bins=val_cutoffs)[0]
        col_labels = ['ct_' + str(i+1) for i in range(nbins)]
        if sequence_library:
            chunk.loc[:,'ct_0'] =  utils.sample(chunk.loc[:,'ct'],int(chunk.loc[:,'ct'].sum()/nbins))
        temp_output_df = pd.concat([chunk,pd.DataFrame(seqs_arr,columns=col_labels)],axis=1)
        col_labels = utils.get_column_headers(temp_output_df)
        #temp_output_df['ct'] = temp_output_df[col_labels].sum(axis=1)      
        temp_output_df.drop('val',axis=1,inplace=True)
        print temp_output_df.shape
        print output_df.shape
        output_df = pd.concat([output_df,temp_output_df],axis=0).copy()
        i = i + 1
    output_df['ct'] = output_df[col_labels].sum(axis=1)   
    output_df.reset_index(inplace=True,drop=True)
    
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
            dtype={'seqs':str,'batch':int},chunksize=args.chunksize)
    else:
        df = pd.io.parsers.read_csv(
            sys.stdin,delim_whitespace=True,
            dtype={'seqs':str,'batch':int},chunksize=args.chunksize)
    
    model_df = io.load_model(args.model)
    output_df = main(
        df,model_df,args.noisemodel,npar,
        nbins,start=args.start,end=args.end,chunksize=args.chunksize)
    
    if args.out:
        outloc = open(args.out,'w')
    else:
        outloc = sys.stdout
    pd.set_option('max_colwidth',int(1e8))

    # Validate dataframe for writting
    output_df = qc.validate_dataset(output_df,fix=True)
    io.write(output_df,outloc)

# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('simulate_sort')
    p.add_argument('-nm', '--noisemodel',
        choices=['LogNormal','Normal','None','Plasmid'],default='Normal',
        help='''Noise Model to use.''')
    p.add_argument(
        '-npar','--noiseparam',default = '[.2]',help = '''
        Parameters for your noise model, as a list. The required parameters are
        LogNormal=[autoflouro,scale],Normal=[scale].''')
    p.add_argument('-m', '--model', default=None,
        help='''
        Filename of Model.
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
    p.add_argument('-cs','--chunksize',default=50000,type=int)
    p.add_argument('-o', '--out', default=None)
    p.set_defaults(func=wrapper)
