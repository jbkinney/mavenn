#!/usr/bin/env python

from __future__ import division
import numpy as np
import argparse
import statsmodels.api as sm
import sys
from subprocess import Popen, PIPE
import sst.nsbestimator as nsb
from sklearn.grid_search import GridSearchCV
if __name__== '__main__':
    import sortseq.utils
from collections import Counter
from cStringIO import StringIO
import pandas as pd
import scipy as sp
import scipy.ndimage
import sst.utils as utils


''' This script estimates MI by implementing a Density Estimation through 
    convolution with a kernel. Other methods are available for other variable 
    types. Currently it appears the 'alternate_calc_MI' is the most reliable.
'''

class KernDE():

    '''Class for kernel density estimation that has a fit and predict method for use in scikit learn GridCV'''

    def __init__(self,bandwidth='scott'): 
        self.bandwidth=bandwidth       

    def fit(self,data):
        self.kde = sm.nonparametric.KDEUnivariate(data)
        self.kde.fit(bw=self.bandwidth,cut=0)

    def score(self,evaldata):
        '''Evaluate log prob of model using the density grid. This method is
           more computationally efficient than using the evaluate method'''
        logprobdens = np.log2(self.kde.density[:-1])
        evalhist,bins = np.histogram(evaldata,bins=self.kde.support)
        logprob = np.sum(logprobdens*evalhist)
        return logprob

    def get_params(self,deep=False):
        param = {'bandwidth':self.bandwidth}
        return param
 
    def set_params(self,**param):
        self.bandwidth=param['bandwidth']   
    
    
def EstimateMI(df):       
        '''Estimate original entropy, this is the only function from this script
            we use for parallel tempering'''   
        
        thekde = KernDE()        
        originalent = 0 #always zero because rank order is flat.
        partialent = 0
        MIvar = 0
        #Find partial entropy for each bin value
        for val in set(df.index):
            tempdf = df.loc[val][0] #separate out all entries of certain bin                           
            thekde.fit(tempdf) #convolve with kernal
            #do reimann sum to find entropy.
            pdens = thekde.kde.density
            dsupport = np.sum(thekde.kde.support.max()-thekde.kde.support.min())/len(thekde.kde.support)
            tent = -np.sum(pdens*np.log2(pdens + 1e-15))*dsupport
            partialent = partialent + len(tempdf)/len(df[0])*tent
            
        MI = originalent - partialent
        V = None #don't know how to calculate this.
        
        
        return MI,V

def alternate_calc_MI(rankexpression,batches):
    
    n_bins = 1000
    n_seqs = len(batches)
    batches = batches - batches.min()
    n_batches = int(batches.max()+1)
    f = sp.zeros((n_batches,n_seqs))
    inds = sp.argsort(rankexpression)
    for i,ind in enumerate(inds):
        f[batches[ind],i] = 1.0/n_seqs # batches are zero indexed

    
    # bin and convolve with Gaussian
    f_binned = sp.zeros((n_batches,n_bins))

    for i in range(n_batches):
        f_binned[i,:] = sp.histogram(f[i,:].nonzero()[0],bins=n_bins,range=(0,n_seqs))[0]
    
    f_reg = scipy.ndimage.gaussian_filter1d(f_binned,0.04*n_bins,axis=1)
    f_reg = f_reg/f_reg.sum()

    # compute marginal probabilities
    p_b = sp.sum(f_reg,axis=1)
    p_s = sp.sum(f_reg,axis=0)

    # finally sum to compute the MI
    MI = 0
    for i in range(n_batches):
        for j in range(n_bins):
            if f_reg[i,j] != 0:
                MI = MI + f_reg[i,j]*sp.log2(f_reg[i,j]/(p_b[i]*p_s[j]))
    return MI

def alt2(df):
    '''This is our standard mutual information calculator which is called when
        lm=mi. It is the fastest currently, but requires building a large matrix
        and so takes a lot of memory and can't run the mpra data set on my computer.'''
    n_bins=1000
    n_seqs = len(df.index)
    
    
    binheaders = utils.get_column_headers(df)
    ct_vec = np.array(df['ct'],dtype=int)
    
    n_batches = len(binheaders)
    '''Create a huge matrix with each column equal to an instance of the (normalized 
        such that sum of column equals 1) sequence. The total number of columns
        that each sequence gets is equal to its total number of counts.''' 
    f = np.repeat(np.array(df[binheaders]),ct_vec,axis=0)
    zlen = f.shape[0]
    #do a cumulative sum
    f_binned = f.cumsum(axis=0)
    #Now we are going to bin. First find bin edges.
    bins = np.linspace(zlen/1000-1,zlen-1,1000,dtype=int)
    '''We want to basically sum our original f matrix within each bin. We currently
        have a cumulative sum of this, so first we will select the entries at
        each bin edge.'''
    f_binned = f_binned[bins,:]
    #subtract off previous entry to get only summation within each range
    f_binned[1:,:] = np.subtract(f_binned[1:,:],f_binned[:-1,:])
    #convolve with gaussian
    f_reg = scipy.ndimage.gaussian_filter1d(f_binned,0.04*n_bins,axis=0)
    #regularize
    f_reg = f_reg/f_reg.sum()

    # compute marginal probabilities
    p_b = sp.sum(f_reg,axis=1)
    p_s = sp.sum(f_reg,axis=0)

    # finally sum to compute the MI
    MI = 0
    for j in range(n_batches):
        for i in range(n_bins):
            if f_reg[i,j] != 0:
                MI = MI + f_reg[i,j]*sp.log2(f_reg[i,j]/(p_b[i]*p_s[j]))
    return MI


'''test out integration method for finding mutual information
#,cum_vec,df,batch_name,n_bins=1000'''
def integrand_1(x):
    #y = global
    index = np.searchsorted(cum_vec,x,side='left')
    return np.array(df.loc[index,batch_name])

def integrand_2(y):
    index = np.searchsorted(cum_vec,y,side='left')
    ans = quadgl(integrand_1,[0,n_seqs])
    return ans*mp.log(ans/pb2/pj[index])
    
def integrator_solve(df):
    cum_vec = np.array(np.cumsum(df['ct']))
    binheaders = utils.get_column_headers(df)
    n_bins = 1000
    n_batches = len(binheaders)
    f_binned = sp.zeros((n_batches,n_bins))
    bins = np.linspace(cum_vec[-1]/1000-1,cum_vec[-1]-1,1000,dtype=int)
    for i in range(n_bins):
         for j in range(n_batches):
             batch_name = binheaders[j]
             f_binned[j,i] = scipy.integrate.quad(integrand_1,bins[i],bins[i+1])[0]
    f_reg = scipy.ndimage.gaussian_filter1d(f_binned,0.04*n_bins,axis=0)
    f_reg = f_reg/f_reg.sum()

    # compute marginal probabilities
    p_b = sp.sum(f_reg,axis=1)
    p_s = sp.sum(f_reg,axis=0)

    # finally sum to compute the MI
    MI = 0
    for j in range(n_batches):
        for i in range(n_bins):
            if f_reg[i,j] != 0:
                MI = MI + f_reg[i,j]*sp.log2(f_reg[i,j]/(p_b[i]*p_s[j]))
    return MI
    
def alt3(df):
    '''MI calculator for use when lm=memsaver. This method isn't particularly well tested
        and was not used for any of the analysis in the paper. However, This method doesn't require
        building a matrix, and so can be used with large data sets like the mpra
        set even on my computer, but it is painfully slow. The coding is not
        very refined though, so I think you could really speed this up if you 
        take the time.'''
    n_bins=1000
    n_seqs = len(df.index)
    binheaders = utils.get_column_headers(df)
    n_batches = len(binheaders)
    
    cumvec = np.array(np.cumsum(df['ct']))
    #normalize so that all counts = 1000
    cumvec = 1000*(cumvec/cumvec[-1])
    #get ready to bin. find indexes of bin edges by looking in cumulative vector of counts
    index = np.searchsorted(cumvec,range(n_bins))
    f_binned = sp.zeros((n_bins,n_batches))
    #initialize list
    left_remainder = [0 for q in range(n_batches)]
    left_index = 0
    
    for i in range(0,n_bins-1): 
        #for each sequence which exists fully within a bin, sum it
        main_sum = np.array(df.loc[index[i]:index[i+1]-1,binheaders].sum(axis=0))
        '''For the sequence on that is split by a bin edge on the right, find
            what fraction of that bin is in the left bin'''
        remainder_fraction = (i+1-cumvec[index[i+1]-1])/(cumvec[index[i+1]] - cumvec[index[i+1]-1])
        '''find the total bin counts by adding previously left over values from the last bin,
            (left remainder) to the main sum, and the fraction of the split sequence that
            exists in this bin'''
        f_binned[i,:] = left_remainder + main_sum + remainder_fraction*df.loc[index[i+1]-1,binheaders]
        #calculate the remainder for the next bin.
        left_remainder = np.array((1-remainder_fraction)*df.loc[index[i+1]-1,binheaders])
    #must do last bin separately
    f_binned[n_bins-1,:] = left_remainder + np.array(df.loc[index[-1]:,binheaders].sum(axis=0))
    f_reg = scipy.ndimage.gaussian_filter1d(f_binned,0.04*n_bins,axis=0)
    f_reg = f_reg/f_reg.sum()

    # compute marginal probabilities
    p_b = sp.sum(f_reg,axis=1)
    p_s = sp.sum(f_reg,axis=0)

    # finally sum to compute the MI
    MI = 0
    for j in range(n_batches):
        for i in range(n_bins):
            if f_reg[i,j] != 0:
                MI = MI + f_reg[i,j]*sp.log2(f_reg[i,j]/(p_b[i]*p_s[j]))
    return MI

def main():
    parser = argparse.ArgumentParser(
        description='''Estimate mutual information between two variables''')
    parser.add_argument(
        '-q1','--q1type',choices=['Continuous','Discrete'],default='Discrete',
        help='Data type for first quantity.')
    parser.add_argument(
        '-q2','--q2type',choices=['Continuous','Discrete'],default='Discrete',
        help='Data type for first quantity.')
    parser.add_argument(
        '-k','--kneig',default='6',help='''If you are estimating Continuous
        vs Continuous, you can overwrite default arguments for the Kraskov 
        estimator here. This argument is number of nearest neighbors 
        to use, with 6 as the default.''')
    parser.add_argument(
        '-td','--timedelay',default='1',help='''Kraskov Time Delay, default=1''')
    parser.add_argument(
        '-cv','--crossvalidate',default=False,choices=[True,False],help=
        '''Cross validate Kernel Density Estimate. Default=False''')
    parser.add_argument(
        '-o','--out',default=False,help='''Output location/type, by 
        default it writes to standard output, if a file name is supplied 
        it will write to a text file''')
    args = parser.parse_args()
    
    
    MI,V = EstimateMI(
        quant1,quant2,args.q1type,args.q2type,timedelay = args.timedelay,
        embedding = args.embedding,kneig = args.kneig,cv=args.crossvalidate)
    if args.out:
                outloc = open(args.out,'w')
    else:
                outloc = sys.stdout
    outloc.write('Mutual Info \n')
    outloc.write('%.5s' %MI)
    if (args.q1type == args.q2type and args.q1type == 'Discrete'):
        outloc.write( ' +/- %.5s' %np.sqrt(V))
    outloc.write('\n')
         
if __name__ == '__main__':
    main()          

            
                
            
    
    


