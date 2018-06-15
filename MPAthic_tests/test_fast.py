#!/usr/bin/env python
import time
import mpathic.qc as qc
import mpathic.utils as utils
import mpathic.fast as fast
import numpy as np

seq = 'ACGT'*25000 + 'A'*19
seqtype = 'dna'

print '-----------------------------'
# Test rc

t = time.time()
x = qc.rc(seq)
p_time = time.time()-t
print 'python rc: %f sec to rc one dna seq of length %d'%(p_time,len(seq))

t = time.time()
x = fast.reverse_complement(seq, safe=False)
c_time = time.time()-t
print 'cython rc: %f sec to rc one dna seq of length %d'%(c_time,len(seq))

print '%.1f-fold speedup.'%(p_time/c_time)

print '-----------------------------'
# Test seq2sitelist

site_length = 20
t = time.time()
x = [seq[i:(i+site_length)] for i in range(len(seq)-site_length+1)]
p_time = time.time()-t
print 'python seq2sitelist: %f sec to chop a seq into %d sites'%\
    (p_time,len(x))

t = time.time()
x = fast.seq2sitelist(seq,site_length, safe=False)
c_time = time.time()-t
print 'cython seq2sitelist: %f sec to chop a seq into %d sites'%\
    (c_time,len(x))

print '%.1f-fold speedup.'%(p_time/c_time)

print '-----------------------------'
# Test seq2sitelist rc feature

site_length = 20
t = time.time()
x = fast.seq2sitelist(seq,site_length, safe=False)
y = [fast.reverse_complement(s, safe=False) for s in x] 
p_time = time.time()-t
print 'python, seq2sitelist w/ rc: %f sec to chop seq into %d rc sites'%\
    (p_time,len(x))

t = time.time()
x = fast.seq2sitelist(seq,site_length, rc=True, safe=False)
c_time = time.time()-t
print 'cython, seq2sitelist w/ rc: %f sec to chop seq into %d rc sites'%\
    (c_time,len(x))

print '%.1f-fold speedup.'%(p_time/c_time)

print '-----------------------------'
# Test seqs2array_for_matmodel

sites = fast.seq2sitelist(seq,20)
site_length = len(sites[0])
num_sites = len(sites)

t = time.time()
seq_dict,inv_dict = utils.choose_dict(seqtype,modeltype='MAT')
sitearray_p = np.zeros([num_sites,(site_length*len(seq_dict))],dtype=int)
for i, site in enumerate(sites):
    sitearray_p[i,:] = utils.seq2mat(site,seq_dict).ravel()
p_time = time.time()-t
print 'python, seqs2array_for_matmodel: %f sec to convert %d %s seqs of length %d'%\
    (p_time,num_sites,seqtype,site_length)

t = time.time()
sitearray = fast.seqs2array_for_matmodel(sites,seqtype,safe=False)
c_time = time.time()-t
print 'cython, seqs2array_for_matmodel: %f sec to convert %d %s seqs of length %d'%\
    (c_time,num_sites,seqtype,site_length)

print '%.1f-fold speedup.'%(p_time/c_time)

print '-----------------------------'
# Test seqs2array_for_nbrmodel

sites = fast.seq2sitelist(seq,20)
site_length = len(sites[0])
num_sites = len(sites)

t = time.time()
seq_dict,inv_dict = utils.choose_dict(seqtype,modeltype='NBR')
sitearray_p = np.zeros([num_sites,((site_length-1)*len(seq_dict))],dtype=int)
for i, site in enumerate(sites):
    sitearray_p[i,:] = utils.seq2matpair(site,seq_dict).ravel()
p_time = time.time()-t
print 'python, seqs2array_for_nbrmodel: %f sec to convert %d %s seqs of length %d'%\
    (p_time,num_sites,seqtype,site_length)

t = time.time()
sitearray = fast.seqs2array_for_nbrmodel(sites,seqtype,safe=False)
c_time = time.time()-t
print 'cython, seqs2array_for_nbrmodel: %f sec to convert %d %s seqs of length %d'%\
    (c_time,num_sites,seqtype,site_length)

print '%.1f-fold speedup.'%(p_time/c_time)

