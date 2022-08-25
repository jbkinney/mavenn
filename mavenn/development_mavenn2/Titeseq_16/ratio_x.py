import numpy as np
import pdb
def ratio_x(R,T,S):
    """R- aXb numpy array for the number of reads for seqeunce s, where a is number of antigen concentrations, b is the number of bins.
    T- aXb numpy array total number of reads
    S- aXb numpy array of the number cells sorted
    
    output - 
    x_guess - aXb numpy array of estimates of the fraction of cells with sequence s sorted into each bin at each antigen exposure
    k_guess - estimate of the fraction of population with sequence s
    b - aXb numpy array of coefficients relating sequence counts to sorted cells
    k_std - estimate of the standard deviation of log-transformed fraction of population with sequence s
    """
    b = np.array([(np.sum(sorts)/sorts).tolist() for sorts in S])
    x_guess = (R*(1./T)/b)
    k_guess = np.nansum(x_guess, axis=1)
    df = np.sum(k_guess>0)
    if df > 1:
        k_std = np.exp(np.nanstd(np.log(k_guess[k_guess>0]))/np.sqrt(df))
    else:
        k_std = np.exp(1)
        
    k_guess = np.exp(np.nanmean(np.log(k_guess[k_guess>0])))
    return x_guess, k_guess, b, k_std

