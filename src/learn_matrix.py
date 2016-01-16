#!/usr/bin/env python

'''A script which produces linear energy matrix models for a given data set.'''
from __future__ import division
#Our standard Modules
import argparse
import numpy as np
import scipy as sp
import sys
#Our miscellaneous functions
import pandas as pd
import sst.utils as utils
from sklearn import linear_model
import sst.EstimateMutualInfoforMImax as EstimateMutualInfoforMImax
import pymc
import sst.stepper as stepper

import sst.gauge_fix as gauge_fix

def weighted_std(values,weights):
    '''Takes in a dataframe with seqs and cts and calculates the std'''
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return (np.sqrt(variance))

def add_label(s):
    return 'ct_' + str(s)    

def MaximizeMI_test(
        seq_mat,df,emat_0,db=None,burnin=1000,iteration=30000,thin=10,
        runnum=0): 
    '''Performs MI maximization using pymc, this function is used if the learning
    method lm=mi is used'''
    @pymc.stochastic(observed=True,dtype=int)
    def sequences(value=seq_mat):
        return 0
    @pymc.stochastic(observed=True,dtype=pd.DataFrame)
    def pymcdf(value=df):
        return 0
    @pymc.stochastic(dtype=float)
    def emat(s=sequences,p=pymcdf,value=emat_0):
        '''Evaluate the log likelihood of this model.
        This is calculated according to the method explained in Kinney et al(2010).               
        The log likelihood is the number of sequences multiplied by the 
        mutual information between model predictions and data.'''
        dot = value[:,:,sp.newaxis]*s
        p['val'] = dot.sum(0).sum(0)                    
        df_sorted = p.sort(columns='val')
        df_sorted.reset_index(inplace=True)
        n_seqs = s.shape[2]     
        MI = EstimateMutualInfoforMImax.alt2(df_sorted)
        return n_seqs*MI
    if db:
        dbname = db + '_' + str(runnum) + '.sql'
	M = pymc.MCMC([sequences,pymcdf,emat],db='sqlite',dbname=dbname)
    else:
	M = pymc.MCMC([sequences,pymcdf,emat])
    M.use_step_method(stepper.GaugePreservingStepper,emat)
    M.sample(iteration,thin=thin)
    emat_mean = np.mean(M.trace('emat')[burnin:],axis=0)
    return emat_mean

def MaximizeMI_memsaver(
        seq_mat,df,emat_0,db=None,burnin=1000,iteration=30000,thin=10,
        runnum=0):
    '''Performs MCMC MI maximzation in the case where lm = memsaver'''    
    @pymc.stochastic(observed=True,dtype=int)
    def sequences(value=seq_mat):
        return 0
    @pymc.stochastic(observed=True,dtype=pd.DataFrame)
    def pymcdf(value=df):
        return 0
    @pymc.stochastic(dtype=float)
    def emat(s=sequences,p=pymcdf,value=emat_0):               
        dot = value[:,:,sp.newaxis]*s
        p['val'] = dot.sum(0).sum(0)                    
        df_sorted = p.sort(columns='val')
        df_sorted.reset_index(inplace=True)
        n_seqs = s.shape[2]     
        MI = EstimateMutualInfoforMImax.alt3(df_sorted)
        return n_seqs*MI
    if db:
        dbname = db + '_' + str(runnum) + '.sql'
	M = pymc.MCMC([sequences,pymcdf,emat],db='sqlite',dbname=dbname)
    else:
	M = pymc.MCMC([sequences,pymcdf,emat])
    M.use_step_method(stepper.GaugePreservingStepper,emat)
    M.sample(iteration,thin=thin)
    emat_mean = np.mean(M.trace('emat')[burnin:],axis=0)
    return emat_mean


def Berg_von_Hippel(df,dicttype,foreground=1,background=0,pseudocounts=1):
    '''Learn models using berg von hippel model. The foreground sequences are
         usually bin_1 and background in bin_0, this can be changed via flags.''' 

    #check that the foreground and background chosen columns actually exist.
    columns_to_check = {'ct_' + str(foreground),'ct_' + str(background)}
    if not columns_to_check.issubset(set(df.columns)):
        raise ValueError('Foreground or Background column does not exist!')

    #get counts of each base at each position
    foreground_counts = utils.profile_counts(df,dicttype,bin_k=foreground)   
    background_counts = utils.profile_counts(df,dicttype,bin_k=background)
    binheaders = utils.get_column_headers(foreground_counts)
    #add pseudocounts to each position
    foreground_counts[binheaders] = foreground_counts[binheaders] + pseudocounts
    background_counts[binheaders] = background_counts[binheaders] + pseudocounts
    #normalize to compute frequencies
    foreground_freqs = foreground_counts.copy()
    background_freqs = background_counts.copy()
    foreground_freqs[binheaders] = foreground_freqs[binheaders].div(
        foreground_freqs[binheaders].sum(axis=1),axis=0)
    background_freqs[binheaders] = background_freqs[binheaders].div(
        background_freqs[binheaders].sum(axis=1),axis=0)
    
    output_df = -np.log(foreground_freqs/background_freqs)
    #change column names accordingly (instead of ct_ we want val_)
    rename_dict = {'ct_' + str(inv_dict[i]):'val_' + str(inv_dict[i]) for i in range(len(seq_dict))}
    output_df = output_df.rename(columns=rename_dict)
    return output_df


def Compute_Least_Squares(raveledmat,batch,sw,alpha=0):
    '''Ridge regression is the only sklearn regressor that supports sample
        weights, which will make this much faster'''
    clf = linear_model.Ridge(alpha=alpha)
    clf.fit(raveledmat,batch,sample_weight=sw)
    emat = clf.coef_
    return emat
    

def main(
        df,dicttype,lm,modeltype='LinearEmat',LS_means_std=None,LS_iterations=4,
        db=None,iteration=30000,burnin=1000,thin=10,runnum=0
        ,initialize='Rand',start=0,end=None,foreground=1,
        background=0,alpha=0,pseudocounts=1,test=False):
    
    seq_dict,inv_dict = utils.choose_dict(dicttype,modeltype=modeltype)
    
    '''Check to make sure the chosen dictionary type correctly describes
         the sequences. An issue with this test is that if you have DNA sequence
         but choose a protein dictionary, you will still pass this test bc A,C,
         G,T are also valid amino acids'''
    lin_seq_dict,lin_inv_dict = utils.choose_dict(dicttype,modeltype='LinearEmat')
    def check_sequences(s):
        return set(s).issubset(lin_seq_dict)
    if False in set(df.seq.apply(check_sequences)):
        raise TypeError('Wrong sequence type!')
    par_seq_dict = {v:k for v,k in seq_dict.items() if k != (len(seq_dict)-1)}
    #select target sequence region
    df.loc[:,'seq'] = df.loc[:,'seq'].str.slice(start,end)
    df = utils.collapse_further(df)
    col_headers = utils.get_column_headers(df)
    df[col_headers] = df[col_headers].astype(int)
    # The different learning methods produce different formats for the emat
    val_cols = ['val_' + inv_dict[i] for i in range(len(seq_dict))]
    if end:
        df = df[df['seq'].apply(len) == (end-start)]
    df.reset_index(inplace=True,drop=True)
    #Drop any sequences with incorrect length
    if not end:
        seqL = len(df['seq'][0]) - start
    else:
        seqL = end-start
    df = df[df.seq.apply(len) == (seqL)]
    #If there were sequences of different lengths, then print error
    if len(set(df.seq.apply(len))) > 1:
         sys.stderr.write('Lenghts of all sequences are not the same!')
    df.reset_index(inplace=True,drop=True)
    #Do something different for each type of learning method (lm)
    if lm == 'bvh':
        emat = Berg_von_Hippel(
            df,dicttype,foreground=foreground,background=background,
            pseudocounts=pseudocounts)
    if lm == 'leastsq':
        
        if LS_means_std: #If user supplied preset means and std for each bin
            means_std_df = pd.io.parsers.read_csv(LS_means_std,delim_whitespace=True)
            #change bin number to 'ct_number' and then use as index
            labels = list(means_std_df['bin'].apply(add_label))
            std = means_std_df['std']
            std.index = labels
            #Change Weighting of each sequence by dividing counts by bin std
            df[labels] = df[labels].div(std)
            means = means_std_df['mean']
            means.index = labels
        else:
            means = None
        
        df['ct'] = df[col_headers].sum(axis=1)
        df = df[df.ct != 0]        
        df.reset_index(inplace=True,drop=True)     
        try:
            df.drop('ct_0',inplace=True)
        except:
            pass
                           
        raveledmat,batch,sw = utils.genweightandmat(
                                  df,par_seq_dict,means=means,modeltype=modeltype)
        #Use ridge regression to find matrix.       
        emat = Compute_Least_Squares(raveledmat,batch,sw,alpha=alpha)

        
    
    if lm == 'mi':
        if initialize == 'Rand':
            if modeltype == 'LinearEmat':
                emat_0 = utils.RandEmat(len(df['seq'][0]),len(seq_dict))
            elif modeltype == 'Neighbor':
                emat_0 = utils.RandEmat(len(df['seq'][0])-1,len(seq_dict)**2)
        elif initialize == 'LeastSquares':
            emat_0_df = main(df.copy(),dicttype,'leastsq',modeltype=modeltype,alpha=alpha)
            emat_0 = np.transpose(np.array(emat_0_df[val_cols]))
        df['ct'] = df[col_headers].sum(axis=1)
        #normalize such that the total counts in each bin sum to 1
        df[col_headers] = df[col_headers].div(df['ct'],axis=0)
        if modeltype == 'Neighbor':
            #choose starting point for MCMC
            seq_mat = np.zeros([len(seq_dict),len(df['seq'][0])-1,len(df.index)],dtype=int)
            for i in range(len(df.index)):
                seq_mat[:,:,i] = utils.seq2matpair(df['seq'][i],seq_dict)
        elif modeltype == 'LinearEmat':
            
            seq_mat = np.zeros([len(seq_dict),len(df['seq'][0]),len(df.index)],dtype=int)
            for i in range(len(df.index)):
                seq_mat[:,:,i] = utils.seq2mat(df['seq'][i],seq_dict)
            #pymc doesn't take sparse mat        
        emat = MaximizeMI_test(
                seq_mat,df,emat_0,db=db,iteration=iteration,burnin=burnin,
                thin=thin,runnum=runnum)
    if lm == 'memsaver':
        if initialize == 'Rand':
            if modeltype == 'LinearEmat':
                emat_0 = utils.RandEmat(len(df['seq'][0]),len(seq_dict))
            elif modeltype == 'Neighbor':
                emat_0 = utils.RandEmat(len(df['seq'][0])-1,len(seq_dict)**2)
        elif initialize == 'LeastSquares':
            emat_0_df = main(df.copy(),dicttype,'leastsq',modeltype=modeltype,alpha=alpha)
            emat_0 = np.transpose(np.array(emat_0_df[val_cols]))
        df['ct'] = df[col_headers].sum(axis=1)
        if modeltype == 'Neighbor':
            #choose starting point for MCMC
            seq_mat = np.zeros([len(seq_dict),len(df['seq'][0])-1,len(df.index)],dtype=int)
            for i in range(len(df.index)):
                seq_mat[:,:,i] = utils.seq2matpair(df['seq'][i],seq_dict)
        elif modeltype == 'LinearEmat':
            
            seq_mat = np.zeros([len(seq_dict),len(df['seq'][0]),len(df.index)],dtype=int)
            for i in range(len(df.index)):
                seq_mat[:,:,i] = utils.seq2mat(df['seq'][i],seq_dict)
            #pymc doesn't take sparse mat        
        emat = MaximizeMI_memsaver(
                seq_mat,df,emat_0,db=db,iteration=iteration,burnin=burnin,
                thin=thin,runnum=runnum)
    if (lm == 'mi' or lm == 'memsaver'):       
        if modeltype == 'Neighbor':
             emat_typical = gauge_fix.fix_neighbor(np.transpose(emat))
             #energy = np.sum(utils.seq2matpair(''.join(wtseq),seq_dict)*emat_typical)
        elif modeltype == 'LinearEmat':
             emat_typical = gauge_fix.fix_matrix(np.transpose(emat))
             #energy = np.sum(utils.seq2mat(wtseq,seq_dict)*emat_typical)
    
    elif lm == 'bvh': 
        '''the emat for this format is currently transposed compared to other formats
        it is also already a data frame with columns [pos,freq_...]'''
        emat_cols = ['val_' + inv_dict[i] for i in range(len(seq_dict))]
        emat_typical = emat[emat_cols]
        emat_typical = (gauge_fix.fix_matrix((np.array(emat_typical))))
        #energy = ((utils.seq2mat(wtseq,seq_dict)).transpose()*(emat_typical)).sum().sum()
        
    else:
        emat_typical = utils.emat_typical_parameterization(emat,len(seq_dict))
        
        if modeltype == 'Neighbor':
             emat_typical = gauge_fix.fix_neighbor(np.transpose(emat_typical))

        elif modeltype == 'LinearEmat':
             emat_typical = gauge_fix.fix_matrix(np.transpose(emat_typical))
    
    em = pd.DataFrame(emat_typical)
    em.columns = val_cols
    if modeltype == 'Neighbor':
        pos = pd.Series(range(start,start - 1 + len(df['seq'][0])),name='pos') 
    else:
        pos = pd.Series(range(start,start + len(df['seq'][0])),name='pos')    
    output_df = pd.concat([pos,em],axis=1)
    return output_df

# Define commandline wrapper
def wrapper(args):        
    #Read in inputs
    if args.i:
        df = pd.io.parsers.read_csv(args.i,delim_whitespace=True)
    else:
        df = pd.io.parsers.read_csv(sys.stdin,delim_whitespace=True)
    
    output_df = main(df,args.type,
        args.learningmethod,modeltype=args.modeltype,db=args.db_filename,LS_means_std=args.LS_means_std,
        LS_iterations=args.LS_iterations,iteration=args.numiterations,burnin=args.
        burnin,thin=args.thin,start=args.start,end=args.end,
        runnum=args.runnum,initialize=args.initialize,
        foreground=args.foreground,background=args.background,alpha=args.penalty,pseudocounts=args.pseudocounts)
    #Write outputs
    if args.out:
        outloc = open(args.out,'w')
    else:
        outloc = sys.stdout
    pd.set_option('max_colwidth',int(1e8)) #makes sure seq columns aren't shortened
    output_df.to_string(
        outloc, index=False,col_space=10,float_format=utils.format_string)

# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('learn_matrix')
    p.add_argument(
        '-s','--start',type=int,default=0,
        help ='Position to start your analyzed region')
    p.add_argument(
        '-e','--end',type=int,default = None,
        help='Position to end your analyzed region')
    p.add_argument('--penalty',type=float,default=0,help='Ridge Regression Penalty')
    p.add_argument(
        '-t', '--type', choices=['dna','rna','protein'], default='dna')
    p.add_argument(
        '-lm','--learningmethod',choices=['bvh','leastsq','lasso','mi','memsaver',
        'iterative_LS'],default='leastsq',
        help = '''Algorithm for determining matrix parameters.''')
    p.add_argument(
        '-mt','--modeltype',choices=['LinearEmat','Neighbor'],default='LinearEmat')
    p.add_argument(
        '--pseudocounts',default=1,type=int,help='''pseudocounts to add''')
    p.add_argument(
        '--LS_means_std',default=None,help='''File name containing mean and std
        of each bin for least squares regression. Defaults to bin number and 1
        respectively.''')
    p.add_argument(
        '--LS_iterations',type=int,default=4,
        help='Number of iterations for iterative_LS')
    p.add_argument(
        '-fg','--foreground',default=1,type=int,help='''The sequence bin to use
        as foreground for the berg-von-hippel model''')
    p.add_argument(
        '-bg','--background',default=0,type=int,help='''The sequence bin to use
        as background for the berg-von-hippel model''')
    p.add_argument(
        '--initialize',default='LeastSquares',choices=['Rand','LeastSquares'],
        help='''How to choose starting point for MCMC''')
    p.add_argument(
        '-rn','--runnum',default=0,help='''For multiple runs this will change
        output data base file name''')            
    p.add_argument(
        '-db','--db_filename',default=None,help='''For mi, If you wish to save
        the trace in a database, put the name of the sqlite data base''')
    p.add_argument(
        '-iter','--numiterations',type = int,default=30000,
        help='''For mi, Number of MCMC iterations''')
    p.add_argument(
        '-b','--burnin',type = int, default=1000,
        help='For mi, Number of burn in iterations')
    p.add_argument(
        '-th','--thin',type=int,default=10,help='''For mi, this option will 
        set the number of iterations during which only 1 iteration 
        will be saved.''')
    p.add_argument(
        '-i','--i',default=False,help='''Read input from file instead 
        of stdin''')
    p.add_argument('-o', '--out', default=None)
    p.set_defaults(func=wrapper)
