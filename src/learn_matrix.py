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
import sortseq_tools.utils as utils
from sklearn import linear_model
import sortseq_tools.EstimateMutualInfoforMImax as EstimateMutualInfoforMImax
import pymc
import sortseq_tools.stepper as stepper
import os
from sortseq_tools import SortSeqError
import sortseq_tools.io as io
import sortseq_tools.gauge_fix as gauge_fix
import sortseq_tools.qc as qc
import pdb
from sortseq_tools import shutthefuckup
import sortseq_tools.numerics as numerics

def weighted_std(values,weights):
    '''Takes in a dataframe with seqs and cts and calculates the std'''
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return (np.sqrt(variance))

def add_label(s):
    return 'ct_' + str(s)    

def MaximizeMI_memsaver(
        seq_mat,df,emat_0,wtrow,db=None,burnin=1000,iteration=30000,thin=10,
        runnum=0,verbose=False):
    '''Performs MCMC MI maximzation in the case where lm = memsaver'''    
    '''
    @pymc.stochastic(observed=True,dtype=sp.sparse.csr_matrix)
    def sequences(value=seq_mat):
        return 0
    '''
    n_seqs = seq_mat.shape[0]
    @pymc.stochastic(observed=True,dtype=pd.DataFrame)
    def pymcdf(value=df):
        return 0
    @pymc.stochastic(dtype=float)
    def emat(p=pymcdf,value=emat_0):         
        p['val'] = numerics.eval_modelmatrix_on_mutarray(np.transpose(value),seq_mat,wtrow)                     
        MI = EstimateMutualInfoforMImax.alt4(p.copy())  # New and improved
        return n_seqs*MI
    if db:
        dbname = db + '_' + str(runnum) + '.sql'
        M = pymc.MCMC([pymcdf,emat],db='sqlite',dbname=dbname)
    else:
        M = pymc.MCMC([pymcdf,emat])
    M.use_step_method(stepper.GaugePreservingStepper,emat)

    if not verbose:
        M.sample = shutthefuckup(M.sample)

    M.sample(iteration,thin=thin)
    emat_mean = np.mean(M.trace('emat')[burnin:],axis=0)
    return emat_mean


def Berg_von_Hippel(df,dicttype,foreground=1,background=0,pseudocounts=1):
    '''Learn models using berg von hippel model. The foreground sequences are
         usually bin_1 and background in bin_0, this can be changed via flags.''' 
    seq_dict,inv_dict = utils.choose_dict(dicttype)
    #check that the foreground and background chosen columns actually exist.
    columns_to_check = {'ct_' + str(foreground),'ct_' + str(background)}
    if not columns_to_check.issubset(set(df.columns)):
        raise SortSeqError('Foreground or Background column does not exist!')

    #get counts of each base at each position
    foreground_counts = utils.profile_counts(df,dicttype,bin_k=foreground)   
    background_counts = utils.profile_counts(df,dicttype,bin_k=background)
    binheaders = utils.get_column_headers(foreground_counts)
    #add pseudocounts to each position
    foreground_counts[binheaders] = foreground_counts[binheaders] + pseudocounts
    background_counts[binheaders] = background_counts[binheaders] + pseudocounts
    #make sure there are no zeros in counts after addition of pseudocounts
    ct_headers = utils.get_column_headers(foreground_counts)
    if foreground_counts[ct_headers].isin([0]).values.any():
        raise SortSeqError('''There are some bases without any representation in\
            the foreground data, you should use pseudocounts to avoid failure \
            of the learning method''')
    if background_counts[ct_headers].isin([0]).values.any():
        raise SortSeqError('''There are some bases without any representation in\
            the background data, you should use pseudocounts to avoid failure \
            of the learning method''')
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
    

def main(df,lm='IM',modeltype='MAT',LS_means_std=None,\
    LS_iterations=4,db=None,iteration=30000,burnin=1000,thin=10,\
    runnum=0,initialize='LeastSquares',start=0,end=None,foreground=1,\
    background=0,alpha=0,pseudocounts=1,test=False,drop_library=False,\
    verbose=False):
    
    # Determine dictionary
    seq_cols = qc.get_cols_from_df(df,'seqs')
    if not len(seq_cols)==1:
        raise SortSeqError('Dataframe has multiple seq cols: %s'%str(seq_cols))
    dicttype = qc.colname_to_seqtype_dict[seq_cols[0]]

    seq_dict,inv_dict = utils.choose_dict(dicttype,modeltype=modeltype)
    
    '''Check to make sure the chosen dictionary type correctly describes
         the sequences. An issue with this test is that if you have DNA sequence
         but choose a protein dictionary, you will still pass this test bc A,C,
         G,T are also valid amino acids'''
    #set name of sequences column based on type of sequence
    type_name_dict = {'dna':'seq','rna':'seq_rna','protein':'seq_pro'}
    seq_col_name = type_name_dict[dicttype]
    lin_seq_dict,lin_inv_dict = utils.choose_dict(dicttype,modeltype='MAT')
    #wtseq = utils.profile_counts(df.copy(),dicttype,return_wtseq=True,start=start,end=end)
    #wt_seq_dict_list = [{inv_dict[np.mod(i+1+seq_dict[w],len(seq_dict))]:i for i in range(len(seq_dict)-1)} for w in wtseq]
    par_seq_dict = {v:k for v,k in seq_dict.items() if k != (len(seq_dict)-1)}
    #drop any rows with ct = 0
    df = df[df.loc[:,'ct'] != 0]
    df.reset_index(drop=True,inplace=True)
    
    #If there are sequences of different lengths, then print error but continue
    if len(set(df[seq_col_name].apply(len))) > 1:
         sys.stderr.write('Lengths of all sequences are not the same!')
    #select target sequence region
    df.loc[:,seq_col_name] = df.loc[:,seq_col_name].str.slice(start,end)
    df = utils.collapse_further(df)
    col_headers = utils.get_column_headers(df)
    #make sure all counts are ints
    df[col_headers] = df[col_headers].astype(int)
    #create vector of column names
    val_cols = ['val_' + inv_dict[i] for i in range(len(seq_dict))]
    df.reset_index(inplace=True,drop=True)
    #Drop any sequences with incorrect length
    if not end:
        '''is no value for end of sequence was supplied, assume first seq is
            correct length'''
        seqL = len(df[seq_col_name][0]) - start
    else:
        seqL = end-start
    df = df[df[seq_col_name].apply(len) == (seqL)]
    df.reset_index(inplace=True,drop=True)
    #Do something different for each type of learning method (lm)
    if lm == 'ER':
        emat = Berg_von_Hippel(
            df,dicttype,foreground=foreground,background=background,
            pseudocounts=pseudocounts)
    if lm == 'LS':
        '''First check that is we don't have a penalty for ridge regression,
            that we at least have all possible base values so that the analysis
            will not fail'''
        if alpha == 0:
            df_counts = utils.profile_counts(df.copy(),dicttype)
            ct_headers = utils.get_column_headers(df_counts)
            if df_counts[ct_headers].isin([0]).values.any():
                raise SortSeqError('''There are some bases without any \
                representation in the data, you should use a ridge regression\
                penalty (specified by --penalty) \
                to avoid failure of the learning method''')
        if LS_means_std: #If user supplied preset means and std for each bin
            means_std_df = io.load_meanstd(LS_means_std)

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
        #drop all rows without counts
        df['ct'] = df[col_headers].sum(axis=1)
        df = df[df.ct != 0]        
        df.reset_index(inplace=True,drop=True)
        ''' For sort-seq experiments, bin_0 is library only and isn't the lowest
            expression even though it is will be calculated as such if we proceed.
            Therefore is drop_library is passed, drop this column from analysis.'''
        if drop_library:
            try:     
                df.drop('ct_0',inplace=True)
                col_headers = utils.get_column_headers(df)
                if len(col_headers) < 2:
                    raise SortSeqError(
                        '''After dropping library there are no longer enough 
                        columns to run the analysis''')
            except:
                raise SortSeqError('''drop_library option was passed, but no ct_0
                    column exists''')
        #parameterize sequences into 3xL vectors
                               
        raveledmat,batch,sw = utils.genweightandmat(
                                  df,par_seq_dict,dicttype,means=means,modeltype=modeltype)
        #Use ridge regression to find matrix.       
        emat = Compute_Least_Squares(raveledmat,batch,sw,alpha=alpha)

    if lm == 'IM':
        seq_mat,wtrow = numerics.dataset2mutarray(df.copy(),modeltype)
        #this is also an MCMC routine, do the same as above.
        if initialize == 'Rand':
            if modeltype == 'MAT':
                emat_0 = utils.RandEmat(len(df[seq_col_name][0]),len(seq_dict))
            elif modeltype == 'NBR':
                emat_0 = utils.RandEmat(len(df['seq'][0])-1,len(seq_dict))
        elif initialize == 'LeastSquares':
            emat_cols = ['val_' + inv_dict[i] for i in range(len(seq_dict))]
            emat_0_df = main(df.copy(),lm='LS',modeltype=modeltype,alpha=alpha,start=0,end=None,verbose=verbose)
            emat_0 = np.transpose(np.array(emat_0_df[emat_cols]))   
            #pymc doesn't take sparse mat        
        emat = MaximizeMI_memsaver(
                seq_mat,df.copy(),emat_0,wtrow,db=db,iteration=iteration,burnin=burnin,
                thin=thin,runnum=runnum,verbose=verbose)
    #now format the energy matrices to get them ready to output
    if (lm == 'IM' or lm == 'memsaver'):       
        if modeltype == 'NBR':
             emat_typical = gauge_fix.fix_neighbor(np.transpose(emat))
        elif modeltype == 'MAT':
             emat_typical = gauge_fix.fix_matrix(np.transpose(emat))
    
    elif lm == 'ER': 
        '''the emat for this format is currently transposed compared to other formats
        it is also already a data frame with columns [pos,val_...]'''
        emat_cols = ['val_' + inv_dict[i] for i in range(len(seq_dict))]
        emat_typical = emat[emat_cols]
        emat_typical = (gauge_fix.fix_matrix((np.array(emat_typical))))
        
    else: #must be Least squares
        emat_typical = utils.emat_typical_parameterization(emat,len(seq_dict))        
        if modeltype == 'NBR':
             emat_typical = gauge_fix.fix_neighbor(np.transpose(emat_typical))
        elif modeltype == 'MAT':
             emat_typical = gauge_fix.fix_matrix(np.transpose(emat_typical))
    
    em = pd.DataFrame(emat_typical)
    em.columns = val_cols
    #add position column
    if modeltype == 'NBR':
        pos = pd.Series(range(start,start - 1 + len(df[seq_col_name][0])),name='pos') 
    else:
        pos = pd.Series(range(start,start + len(df[seq_col_name][0])),name='pos')    
    output_df = pd.concat([pos,em],axis=1)

    # Validate model and return
    output_df = qc.validate_model(output_df,fix=True)
    return output_df

# Define commandline wrapper
def wrapper(args):

    #validate some of the input arguments
    qc.validate_input_arguments_for_learn_matrix(
        foreground=args.foreground,background=args.background,
        modeltype=args.modeltype,learningmethod=args.learningmethod,
        start=args.start,end=args.end,iteration=args.iteration,
        burnin=args.burnin,thin=args.thin,pseudocounts=args.pseudocounts,)

    inloc = io.validate_file_for_reading(args.i) if args.i else sys.stdin
    input_df = io.load_dataset(inloc)
    
    outloc = io.validate_file_for_writing(args.out) if args.out else sys.stdout
    #pdb.set_trace()

    output_df = main(input_df,lm=args.learningmethod,\
        modeltype=args.modeltype,db=args.db_filename,\
        LS_means_std=args.LS_means_std,\
        LS_iterations=args.LS_iterations,iteration=args.iteration,\
        burnin=args.burnin,thin=args.thin,start=args.start,end=args.end,\
        runnum=args.runnum,initialize=args.initialize,\
        foreground=args.foreground,background=args.background,\
        alpha=args.penalty,pseudocounts=args.pseudocounts,
        verbose=args.verbose)

    io.write(output_df,outloc)


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
        '-lm','--learningmethod',choices=['ER','LS','lasso','IM',
        'iterative_LS'],default='LS',
        help = '''Algorithm for determining matrix parameters.''')
    p.add_argument(
        '-mt','--modeltype', choices=['MAT','NBR'], default='MAT')
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
        '-db','--db_filename',default=None,help='''For IM, If you wish to save
        the trace in a database, put the name of the sqlite data base''')
    p.add_argument(
        '-dl','--drop_library',default=False,action='store_true',help='''If
        you sorted your library into bin_0, and wish to do least squares analysis
        you should use this option.''')
    p.add_argument(
        '-iter','--iteration',type = int,default=30000,
        help='''For IM, Number of MCMC iterations''')
    p.add_argument(
        '-b','--burnin',type = int, default=1000,
        help='For IM, Number of burn in iterations')
    p.add_argument(
        '-th','--thin',type=int,default=10,help='''For IM, this option will 
        set the number of iterations during which only 1 iteration 
        will be saved.''')
    p.add_argument(
        '-i','--i',default=False,help='''Read input from file instead 
        of stdin''')
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('-o', '--out', default=None)
    p.set_defaults(func=wrapper)
