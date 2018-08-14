#!/usr/bin/env python
'''Class container for Model Handling in the sortseq package. We currently
    only support linear and nearest neighbor models for use within the package.
    In the future, support for additional model types will be added'''
from __future__ import division
import importlib
import sys
import numpy as np
#import utils as utils
from mpathic.src import utils
import scipy as sp
import pandas as pd
import pdb
#import qc as qc
from mpathic.src import qc
#import io_local as io
from mpathic.src import io_local as io

#import fast
from mpathic.src import fast

#from setuptools import Extension
#Extension("fast",["fast.c"])

import time
#from . import SortSeqError
#from src.__init__ import SortSeqError
#from __init__ import SortSeqError
from mpathic import SortSeqError
#import numerics as numerics
from mpathic.src import numerics


class ExpModel:
    #All exp models return log expression
    pass


class LinearModel(ExpModel):
    ''' This model generates energies of binding 
        through a linear energy matrix model'''

    def __init__(self,model_df):
        """
        Constructor takes model parameters in the form of a model dataframe
        """
        model_df = qc.validate_model(model_df.copy(),fix=True)
        seqtype, modeltype = qc.get_model_type(model_df)
        if not modeltype=='MAT':
            raise SortSeqError('Invalid modeltype: %s'%modeltype)

        seq_dict,inv_dict = utils.choose_dict(seqtype,modeltype=modeltype)
        self.seqtype = seqtype
        self.seq_dict = seq_dict
        self.inv_dict = inv_dict
        self.df = model_df
        self.length = model_df.shape[0]

        # Extract matrix part of model dataframe
        headers = qc.get_cols_from_df(model_df,'vals')
        self.matrix = np.transpose(np.array(model_df[headers]))

    def genexp(self,seqs):
        return self.evaluate(seqs)

    def evaluate(self,seqs):
        # Check seqs container
        if isinstance(seqs,pd.DataFrame):
            seq_col = qc.get_cols_from_df(seqs,'seqs')[0]
            seqs_to_use = list(seqs[seq_col])
        elif not (isinstance(seqs,list) or isinstance(seqs,pd.Series)):
            raise SortSeqError('Sequences must be input as a list, pd.Series, or pd.DataFrame')
        else:
            seqs_to_use = list(seqs)

        # Check length
        if len(seqs_to_use[0]) != self.length:
            raise SortSeqError(\
                'Energy Matrix Length does not equal Sequence Length')

        # Compute seqmats
        t0 = time.time()

        # fast.seqs2array_for_matmodel expects seqtype to be bytes
        #self.seqtype = str.encode(self.seqtype) -> type bytes
        #print(self.seqtype)
        #self.seqtype = str.encode(self.seqtype)
        #self.seqtype = str(self.seqtype).encode()
        #print('In Models...')
        #print(type(self.seqtype))
        # if not bytes, change to bytes
        if not (isinstance(self.seqtype,bytes)):
            self.seqtype = str(self.seqtype).encode('utf-8')
        #print(type(self.seqtype))

        #print(type(self.seqtype))
        #print(qc.seqtypes)
        #seqs_to_use = list(map(bytes, str(seqs_to_use).encode('UTF-8')))
        #seqs_to_use = list(map(bytes, seqs_to_use))
        #print(seqs_to_use)

        #if (isinstance(seqs_to_use[0], bytes)):
            #print(seqs_to_use[0].decode())
        #    for i in range(len(seqs_to_use)):
        #        seqs_to_use[i] = seqs_to_use[i].decode()

        # change elements to bytes if they're not bytes
        if not (isinstance(seqs_to_use[0],bytes)):
            #print('changing seq to bytes...')
            for i in range(len(seqs_to_use)):
                seqs_to_use[i] = str(seqs_to_use[i]).encode('utf-8')

        #print('calling cython:')
        #seqarray = fast.seqs2array_for_matmodel(list(seqs_to_use),self.seqtype)
        seqarray = fast.seqs2array_for_matmodel(seqs_to_use, self.seqtype)
        t1 = time.time()

        # Compute and return values
        vals = self.evaluate_on_seqarray(seqarray)
        t2 = time.time()

        #print 't1-t0 = %.4f, t1-t2 = %.4f'%(t1-t0,t2-t1)
        return vals 

    def evaluate_on_seqarray(self, seqarray):
        matrix_vec = np.matrix(np.array(self.matrix.T).ravel()).T
        return np.array(np.matrix(seqarray)*matrix_vec).ravel()

    def evaluate_on_mutarray(self, mutarray, wtrow):
        return numerics.eval_modelmatrix_on_mutarray(\
            modelmatrix=self.matrix.T, mutarray=mutarray, wtrow=wtrow)

class NeighborModel(ExpModel):
    ''' This model generates energies of binding 
        through a nearest neighbor matrix model'''

    def __init__(self,model_df):
        """
        Constructor takes model parameters in the form of a model dataframe
        """
        model_df = qc.validate_model(model_df.copy(),fix=True)
        seqtype, modeltype = qc.get_model_type(model_df)
        if not modeltype=='NBR':
            raise SortSeqError('Invalid modeltype: %s'%modeltype)

        seq_dict,inv_dict = utils.choose_dict(seqtype,modeltype=modeltype)
        self.seqtype = seqtype
        self.seq_dict = seq_dict
        self.inv_dict = inv_dict
        self.df = model_df
        self.length = model_df.shape[0]+1

        # Extract matrix part of model dataframe
        headers = qc.get_cols_from_df(model_df,'vals')
        self.matrix = np.transpose(np.array(model_df[headers]))

    def genexp(self,seqs):
        return self.evaluate(seqs)

    def evaluate(self,seqs):
        # Check seqs container
        if isinstance(seqs,pd.DataFrame):
            seq_col = qc.get_cols_from_df(seqs,'seqs')[0]
            seqs_to_use = list(seqs[seq_col])
        elif not (isinstance(seqs,list) or isinstance(seqs,pd.Series)):
            raise SortSeqError('Sequences must be input as a list, pd.Series, or pd.DataFrame')
        else:
            seqs_to_use = list(seqs)

        # Check length
        if len(seqs_to_use[0]) != self.length:
            raise SortSeqError(\
                'Energy Matrix Length does not equal Sequence Length')

        # Compute seqmats
        t0 = time.time()
        # python 3 fast.c update for string to bytes conversion

        # if not bytes, change to bytes
        if not (isinstance(self.seqtype,bytes)):
            self.seqtype = str(self.seqtype).encode('utf-8')

        # change elements to bytes if they're not bytes
        if not (isinstance(seqs_to_use[0],bytes)):
            #print('changing seq to bytes...')
            for i in range(len(seqs_to_use)):
                seqs_to_use[i] = str(seqs_to_use[i]).encode('utf-8')

        seqarray = fast.seqs2array_for_nbrmodel(seqs_to_use,self.seqtype)
        t1 = time.time()

        # Compute and return values
        vals = self.evaluate_on_seqarray(seqarray)
        t2 = time.time()

        return vals 

    def evaluate_on_seqarray(self, seqarray):
        matrix_vec = np.matrix(np.array(self.matrix.T).ravel()).T
        return np.array(np.matrix(seqarray)*matrix_vec).ravel()

    def evaluate_on_mutarray(self, mutarray, wtrow):
        return numerics.eval_modelmatrix_on_mutarray(\
            modelmatrix=self.matrix.T, mutarray=mutarray, wtrow=wtrow)


class RaveledModel(ExpModel):
    '''This model type generates an energy matrix model, 
        but accepts flattened matrices (for sklearn based fitting). It will not
        generate models from files or dataframes.'''

    def __init__(self,param):
          self.matrix = param
     
    def genexp(self,sparse_seqs_mat,sample_weight=None):
        if sp.sparse.issparse(sparse_seqs_mat):
            '''If modeltype is an energy matrix for repression or activation,
                this will calculate the binding energy of a sequence, which will
                be monotonically correlated with expression.'''
            n_seqs = sparse_seqs_mat.shape[0]
            #energies = sp.zeros(n_seqs) 
            energies = np.zeros(n_seqs)
            energiestemp = sparse_seqs_mat.multiply(self.matrix).sum(axis=1)
            for i in range(0,n_seqs):
                 energies[i] = energiestemp[i]
        else:
            raise SortSeqError('Enter sparse sequence matrix.')
        if sample_weight:
            t_exp = np.zeros(np.sum(sample_weights))
            counter=0
            for i, sw in enumerate(sample_weight):
                t_exp[counter:counter+sample_weight[i]] = -energies[i]
                counter = counter + sample_weight[i]
            return t_exp
        else:
            return energies

class NoiseModel():

    def gennoisyexp(self,logexp):
        '''If no particular noise function is declared, then just assume no
            noise'''
        return logexp

    def genlist(self,df):

        nlist = []
        T_counts = np.sum(df['ct'])
        nexp = np.zeros(T_counts)
        counter = 0
        seqcol = [c for c in df.columns if qc.is_col_type(c,'seqs')][0]
        for i,seq in enumerate(df[seqcol]):
            exp = df['val'][i]
            counts = df['ct'][i]
            explist = [exp for z in range(counts)]
            noisyexp = self.gennoisyexp(explist)
            nlist.append(noisyexp)
            nexp[counter:counter+counts]=noisyexp
            counter = counter + counts
        return nexp,nlist
    
        

class LogNormalNoise(NoiseModel):
    '''Noise model that adds autoflourescence and then draws the noisy 
        measurement from a log normal distribution with this mean'''

    def __init__(self,npar):
        self.auto = float(npar[0])
        self.scale = float(npar[1])
        if self.auto <=0 or self.scale <= 0:
            raise SortSeqError('Noise model parameters must be great than zero')

    def gennoisyexp(self,logexp):
        exp = np.exp(logexp) + self.auto
        nexp = np.random.lognormal(np.log(exp),self.scale)
        return np.log(nexp)

#class PlasmidNoise(NoiseModel):
#    '''Noise model that adds autoflourescence and then draws the noisy 
#        measurement from a log normal distribution with this mean'''

#    def __init__(self,npar):
#        R = 10
#        N_mean = 10

#    def gennoisyexp(self,logexp):
#        Ns = np.
#        exp = np.exp(logexp) + self.auto
#        nexp = np.random.lognormal(np.log(exp),self.scale)
#        return np.log(nexp)

class NormalNoise(NoiseModel):
    '''Add Gaussian Noise'''

    def __init__(self,npar):
        try:
            self.scale = float(npar[0])
        except ValueError:
            raise SortSeqError('your input parameter must be a float')
        #Check that scale is in the correct range
        if self.scale <= 0:
            raise SortSeqError('''your input scale for normal noise must be greater\
                than zero''')       

    def gennoisyexp(self,logexp):
        #set scale of normal distribution and add 1e-6 to make sure its not zero
        s = (np.abs(logexp).mean())*self.scale + 1e-6
        #add noise to each entry
        nexp = logexp + np.random.normal(
            scale=s,size=len(logexp))
        return nexp

class PoissonNoise(NoiseModel):
    '''Add Noise for mpra experiment Expression Measurements'''

    def gennoisyexp(self,df,T_LibCounts,T_mRNACounts,mult=1):
        exp = np.exp(-df['val']*mult)
        
        libcounts = df['ct']
        
        weights = exp*libcounts
        meanexp = T_mRNACounts*weights/np.sum(weights)
        
        meanlibcounts = libcounts/np.sum(libcounts)*T_LibCounts
        noisyexp = np.random.poisson(lam=meanexp)
        noisylibcounts = np.random.poisson(lam=meanlibcounts)
        return noisylibcounts,noisyexp

