#!/usr/bin/env python

from __future__ import division
import numpy as np
import argparse
import sys
from subprocess import Popen, PIPE
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
if __name__== '__main__':
    import sortseq.utils
from collections import Counter
from cStringIO import StringIO
import pandas as pd
import scipy as sp
import scipy.ndimage
import utils as utils
import pdb
import info as info


''' This script estimates MI by implementing a Density Estimation through 
    convolution with a kernel. Other methods are available for other variable 
    types. Currently it appears the 'alternate_calc_MI' is the most reliable.
'''

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
    

def alt4(df, coarse_graining_level = 0.01,return_freg=False):
    '''
    MI ESTIMATOR EDITED BY JBK 
    Used when lm=memsaver 
    REQUIRES TESTING AND PROFILING.
    '''
    n_groups=500
    n_seqs = len(df.index)
    binheaders = utils.get_column_headers(df)
    n_batches = len(binheaders)
    cts_grouped = sp.zeros([n_groups,n_batches])
    group_num = 0
    frac_empty = 1.0
    
    #copy dataframe
    tmp_df = df.copy(binheaders+['val'])

    # Speed computation by coarse-graining model predictions
    if coarse_graining_level:
        assert type(coarse_graining_level)==float
        assert coarse_graining_level > 0
        vals = tmp_df['val'].values
        scale = np.std(vals)
        coarse_vals = np.floor((vals/scale)/coarse_graining_level)
        tmp_df['val'] = coarse_vals
        grouped = tmp_df.groupby('val')
        grouped_tmp_df = grouped.aggregate(np.sum)
        grouped_tmp_df.sort_index(inplace=True)
    else:
        grouped_tmp_df = tmp_df
        grouped_tmp_df.sort_values(by='val',inplace=True)
    # Get ct_xxx columns
    ct_df = grouped_tmp_df[binheaders].astype(float)
    cts_per_group = ct_df.sum(axis=0).sum()/n_groups
    # Histogram counts in groups. This is a bit tricky
    group_vec = np.zeros(n_batches)
    for i,row in ct_df.iterrows():
        row_ct_tot = row.sum()
        row_ct_vec = row.values
        row_frac_vec = row_ct_vec/row_ct_tot 

        while row_ct_tot >= cts_per_group*frac_empty:
            group_vec = group_vec + row_frac_vec*(cts_per_group*frac_empty)
            row_ct_tot -= cts_per_group*frac_empty

            # Only do once per group_num
            cts_grouped[group_num,:] = group_vec.copy() 
            # Reset for new group_num
            group_num += 1
            frac_empty = 1.0
            group_vec[:] = 0.0
        group_vec += row_frac_vec*row_ct_tot
        
        frac_empty -= row_ct_tot/cts_per_group
    if group_num == n_groups-1:
        cts_grouped[group_num,:] = group_vec.copy()
    elif group_num == n_groups:
        pass
    else:
        raise TypeError(\
            'group_num=%d does not match n_groups=%s'%(group_num,n_groups))
    # Smooth empirical distribution with gaussian KDE
    f_reg = scipy.ndimage.gaussian_filter1d(cts_grouped,0.08*n_groups,axis=0)

    # Return mutual information
    if return_freg:
       return info.mutualinfo(f_reg),f_reg
    else:
       return info.mutualinfo(f_reg)


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

            
                
            
    
    


