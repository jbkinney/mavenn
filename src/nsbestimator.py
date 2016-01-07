#!/usr/bin/env python

"""
nsb_entropy.py


This script is a python version of Mathematica functions by Christian Mendl, 
with a python implementation by Sungho Hong, and altered by Bill Ireland, 
implementing the Nemenman-Shafee-Bialek (NSB) estimator of entropy.

"""
import sys

from mpmath import psi, rf, power, quadgl, mp
import numpy as np
import scipy as sp

DPS = 20

def make_nxkx(n, K):
  """
  Return the histogram of the input histogram n assuming 
  that the number of all bins is K.
  
  >>> from numpy import array
  >>> nTest = array([4, 2, 3, 0, 2, 4, 0, 0, 2])
  >>> make_nxkx(nTest, 9)
  {0: 3, 2: 3, 3: 1, 4: 2}
  """
  nxkx = {}
  nn = n[n>0]
  unn = np.unique(nn)
  for x in unn:
    nxkx[x] = (nn==x).sum()
  if K>nn.size:
    nxkx[0] = K-nn.size
  return nxkx

def _xi(beta, K): # This is <H_q|B>
  return psi(0, K*beta+1)-psi(0, beta+1)

def _dxi(beta, K): #this is d<H_q|B>/dB
  return K*psi(1, K*beta+1)-psi(1, beta+1)

def _S1i(x, nxkx, beta, N, kappa):
  return nxkx[x] * (x+beta)/(N+kappa) * (psi(0, x+beta+1)-psi(0, N+kappa+1))

def _S1(beta, nxkx, N, K): # this is <H_q|c,b>
  kappa = beta*K
  rx = np.array([_S1i(x, nxkx, beta, N, kappa) for x in nxkx])
  return -rx.sum()

def _rhoi(x, nxkx, beta):
  return power(rf(beta, np.double(x)), nxkx[x])

def _rho(beta, nxkx, N, K):# this is p(B|c)*log(m)/p(B)
  kappa = beta*K
  rx = np.array([_rhoi(x, nxkx, beta) for x in nxkx])
  return rx.prod()/rf(kappa, np.double(N))

def _Si(w, nxkx, N, K): 
  '''This caluclates p(B|c)*<H_q|c,beta> This is the integrand for 
      the nsb mean entropy estimation'''
  sbeta = w/(1-w)
  beta = sbeta*sbeta
  return _rho(beta, nxkx, N, K) * _S1(beta, nxkx, N, K) * _dxi(beta, K) * 2*sbeta/(1-w)/(1-w)

def _measure(w, nxkx, N, K): # This calculates p(Beta|c), integrate this and use this for normalization
  sbeta = w/(1-w)
  beta = sbeta*sbeta
  return _rho(beta, nxkx, N, K) * _dxi(beta, K) * 2*sbeta/(1-w)/(1-w)

def S(x, N, K): 
  """
  Return the estimated entropy. x is a vector of counts, nxkx is the histogram 
  of the input histogram constructed by make_nxkx. N is the total number of 
  elements, and K is the degree of freedom.
  
  >>> from numpy import array
  >>> nTest = array([4, 2, 3, 0, 2, 4, 0, 0, 2])
  >>> K = 9  # which is actually equal to nTest.size.
  >>> S(make_nxkx(nTest, K), nTest.sum(), K)
  1.9406467285026877476
  """
  nxkx = make_nxkx(x,K)
  mp.dps = DPS
  mp.pretty = True
  
  f = lambda w: _Si(w, nxkx, N, K)
  g = lambda w: _measure(w, nxkx, N, K)  
  return np.log2(np.exp(1))*quadgl(f, [0, 1])/quadgl(g, [0, 1])
def nsb_integrand_variance_term1(beta,nxkx,N,m): 
    '''computer variance of the nsb esimator, with term 1,2  
        names refering to eq 3.34 from kinney thesis, the third term, 
        which is simply the square of the expectation value of the
        entropy, is subtracted off at the end'''
    summand1 = sp.array(
        [nxkx[x]*(x + beta+1)*(x+beta)/((N+m*beta+1)*(N+m*beta))*
        psi(1,x + beta + 1) for x in nxkx])
    summand2 = sp.array(
        [nxkx[x]*(x+beta)/((N+m*beta+1)*(N+m*beta))*psi(0,x+beta+1)**2 
        for x in nxkx])
    summand3 = sp.array(
        [nxkx[x]*(x+beta)/(N+m*beta)*psi(0,x+beta+1) for x in nxkx])
    return (
        sp.sum(summand1) - psi(1,N + beta*m+1) + sp.sum(summand2)-1/
        (N+m*beta+1)*sp.sum(summand3)**2)
def nsb_integrand_variance_term2(beta,nxkx,N,m):
    summand1 = sp.array(
        [nxkx[x]*((x+beta)/(N+m*beta)*psi(0,x+beta+1)) for x in nxkx])
    return (psi(0,N+beta*m+1)-sp.sum(summand1))**2
def _dSi(w, nxkx, N, K):
  sbeta = w/(1-w)
  beta = sbeta*sbeta
  return (_rho(beta, nxkx, N, K) * (nsb_integrand_variance_term1(beta,nxkx,N,K)
      + nsb_integrand_variance_term2(beta,nxkx,N,K)) * _dxi(beta, K) 
      * 2*sbeta/(1-w)/(1-w))

def dS(x, N, K):
  """
  Returns the Variance in the entropy
  
  >>> from numpy import array, sqrt
  >>> nTest = np.array([4, 2, 3, 0, 2, 4, 0, 0, 2])
  >>> K = 9  # which is actually equal to nTest.size.
  >>> nxkx = make_nxkx(nTest, K)
  >>> s = S(nxkx, nTest.sum(), K)
  >>> ds = dS(nxkx, nTest.sum(), K)
  >>> ds
  3.7904532836824960524
  >>> sqrt(ds-s**2) # the standard deviation for the estimated entropy.
  0.15602422515209426008
  """  
  nxkx = make_nxkx(x,K)
  mp.dps = DPS
  mp.pretty = True
  
  f = lambda w: _dSi(w, nxkx, N, K)
  g = lambda w: _measure(w, nxkx, N, K)  
  return (
      np.log2(np.exp(1))*quadgl(f, [0, 1])/quadgl(g, [0, 1]) - 
      S(x,N,K)**2/np.log2(np.exp(1)))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
