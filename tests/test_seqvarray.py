#!/usr/bin/env python
import time
import sortseq_tools.simulate_library
import sortseq_tools.fast as fast
import sortseq_tools.qc as qc
from sortseq_tools.profile_mut import main as profile_mut
from sortseq_tools.simulate_library import main as simulate_library
import numpy as np
from scipy.sparse import csr, csr_matrix, lil_matrix
from sortseq_tools import SortSeqError
import pdb
import sys

def nbytes(array):
    if isinstance(array,np.ndarray):
        return array.nbytes
    elif isinstance(array,csr.csr_matrix):
        return array.data.nbytes + array.indptr.nbytes + array.indices.nbytes
    else:
        return sizeof(array)


def dataset2seqarray(dataset_df, modeltype):
    # Determine the type of model and set seq2array function appropriately
    if modeltype=='MAT':
        seqs2array = sortseq_tools.fast.seqs2array_for_matmodel
    elif modeltype=='NBR':
        seqs2array = sortseq_tools.fast.seqs2array_for_nbrmodel
    else:
        raise SortSeqError('Unknown model type: %s'%modeltype)
    seqcol = qc.get_cols_from_df(dataset_df,'seqs')[0]  
    seqtype = qc.colname_to_seqtype_dict[seqcol]
    seqlist = list(dataset_df[seqcol])
    seqarray = seqs2array(seqlist, seq_type=seqtype)
    return seqarray

# Given a large list of sequences, produce a (sparse) mutation array
def dataset2mutarray(dataset_df, modeltype,\
    chunksize=1000, rowsforwtcalc=100, forfitting=True):

    # Determine the type of model and set seq2array function appropriately
    if modeltype=='MAT':
        seqs2array = sortseq_tools.fast.seqs2array_for_matmodel
    elif modeltype=='NBR':
        seqs2array = sortseq_tools.fast.seqs2array_for_nbrmodel
    else:
        raise SortSeqError('Unknown model type: %s'%modeltype)

    # Determine seqtype, etc.
    seqcol = qc.get_cols_from_df(dataset_df,'seqs')[0]
    seqtype = qc.colname_to_seqtype_dict[seqcol]
    wtcol = qc.seqtype_to_wtcolname_dict[seqtype]

    # Compute the wt sequence
    rowsforwtcalc = min(rowsforwtcalc,dataset_df.shape[0])
    dataset_head_df = dataset_df.head(rowsforwtcalc)
    mut_df = profile_mut(dataset_head_df)
    wtseq = ''.join(list(mut_df[wtcol]))
    wtrow = seqs2array([wtseq], seq_type=seqtype).ravel().astype(bool)
    numfeatures = len(wtrow)

    # Process dataframe in chunks
    startrow = 0
    endrow = startrow+chunksize-1
    numrows = dataset_df.shape[0]

    # Fill in mutarray (a lil matrix) chunk by chunk
    mutarray_lil = lil_matrix((numrows,numfeatures),dtype=int)
    matrix_filled = False
    while not matrix_filled:

        if startrow >= numrows:
            matrix_filled = True
            continue
        elif endrow >= numrows:
            endrow = numrows-1
            matrix_filled = True

        # Compute seqarray
        seqlist = list(dataset_df[seqcol][startrow:(endrow+1)])
        seqarray = seqs2array(seqlist, seq_type=seqtype)

        # Remove wt entries
        tmp = seqarray.copy()
        tmp[:,wtrow] = 0

        # Store results from this chunk
        mutarray_lil[startrow:(endrow+1),:] = tmp

        # Increment rows
        startrow = endrow+1
        endrow = startrow + chunksize - 1

    # Convert to csr matrix
    mutarray_csr = mutarray_lil.tocsr()

    # Return vararray as well as binary representation of wt seq
    return mutarray_csr, wtrow


# Create sequences to test this on
wtseq = 'AAAAAAAGTGAGATGGCAATCTAATTCGGCACCCCAGGTTTTACACTTTATGCTTCCGGCTCGTATGTTGTGTGG'
dataset_df = simulate_library(wtseq,numseq=10000,mutrate=.1,tags=True)
seqarray = dataset2seqarray(dataset_df, modeltype='MAT')
mutarray, wtrow = dataset2mutarray(dataset_df, modeltype='MAT')

# Print compression results
seqarray_size = nbytes(seqarray)
mutarray_size = nbytes(mutarray)

print 'size of seqarray = %d'%seqarray_size
print 'size of mutarray = %d'%mutarray_size
print 'compression ratio = %.1f'%(1.*seqarray_size/mutarray_size)