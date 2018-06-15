import numpy as np
cimport numpy as np
from cpython cimport bool

import qc as qc
import re
import scipy as sp
#from . import SortSeqError
#from src.__init__ import SortSeqError
#from __init__ import SortSeqError
from mpathic import SortSeqError


DTYPE = np.int
ctypedef np.int_t DTYPE_t
cdef dict rc_dict = qc.rc_dict


def reverse_complement(bytes dna_str, bool safe=True):
    """
    Reverse-complements a DNA sequence
    """
    # Validate sequence if in safe mode
    if safe and qc.seqerr_re_dict['dna'].search(dna_str):
        raise SortSeqError('Invalid character found in DNA sequence.')

    # Compute rc sequence
    cdef bytes B
    cdef bytes c_str = bytes(b''.join([rc_dict[B] for B in dna_str]))
    return c_str[::-1]


def seq2sitelist(bytes seq, int site_length, safe=True, rc=False):
    """
    Chops a sequence into sites of length site_length
    """
    cdef int seq_length = len(seq)
    cdef int num_sites = seq_length - site_length + 1
    cdef int i
    cdef bytes rcseq

    # Make sure there are actually sites (if in safe mode)
    if safe and num_sites<=0:
        raise SortSeqError('site_length > seq_length.')

    # Initialize site_list
    cdef list site_list = [b'.'*site_length]*num_sites

    # Fill site_list
    if rc:
        rcseq = reverse_complement(seq, safe=safe)
        for i in range(num_sites):
            site_list[i] = rcseq[(seq_length-site_length-i):(seq_length-i)]
    else:
        for i in range(num_sites):
            site_list[i] = seq[i:i+site_length]

    return site_list


def seqs2array_for_matmodel(list seq_list, bytes seq_type, bool safe=True):
    """
    Converts a list of sequences (all of which must be the same length) to a numpy array to be used for matrix model evalution
    """
    cdef np.ndarray[DTYPE_t, ndim=2] mat
    cdef bytes c, seq
    cdef int num_seqs, seq_length, num_chars, i, n, k
    cdef dict c_to_i_dict

    # Validate seq_type if in safe mode
    if safe and (not seq_type in qc.seqtypes):
        raise SortSeqError('Invalid seq_type: %s.'%seq_type)

    # Get character dictionary
    c_to_i_dict = qc.char_to_mat_index_dicts[seq_type]
    num_chars = len(c_to_i_dict)

    # Initialize matrix
    num_seqs = len(seq_list)
    seq_length = len(seq_list[0])
    mat = np.zeros([num_seqs,num_chars*seq_length], dtype=DTYPE)

    # Fill matrix row by row
    for n, seq in enumerate(seq_list):

        # Validate sequence composition if in safe mode
        if safe and qc.seqerr_re_dict[seq_type].search(seq):
            raise SortSeqError(\
                'Invalid character found in %s sequence.'%seq_type)

        # Validate sequence length if in safe mode
        if safe and len(seq)!=seq_length:
            raise SortSeqError('Sequences are not all the same length.')

        # Fill in array
        for i, c in enumerate(seq):
            k = c_to_i_dict[c]
            mat[n,num_chars*i+k] = 1
    return mat
    

def seqs2array_for_nbrmodel(list seq_list, bytes seq_type, safe=True):
    """
    Converts a list of sequences (all of which must be the same length) 
    to a numpy array to be used for neighbor model evalution
    """
    cdef np.ndarray[DTYPE_t, ndim=2] mat
    cdef bytes seq, dichar
    cdef int num_seqs, seq_length, num_dichars, i, n, k
    cdef dict c_to_i_dict

    # Validate seq_type if in safe mode
    if safe and (not seq_type in qc.seqtypes):
        raise SortSeqError('Invalid seq_type: %s.'%seq_type)

    # Get character dictionary
    c_to_i_dict = qc.char_to_nbr_index_dicts[seq_type]
    num_dichars = len(c_to_i_dict)

    # Initialize matrix
    num_seqs = len(seq_list)
    seq_length = len(seq_list[0])
    mat = np.zeros([num_seqs,num_dichars*(seq_length-1)], dtype=DTYPE)

    # Fill matrix row by row
    for n, seq in enumerate(seq_list):

        # Validate sequence composition if in safe mode
        if safe and qc.seqerr_re_dict[seq_type].search(seq):
            raise SortSeqError(\
                'Invalid character found in %s sequence.'%seq_type)

        # Validate sequence length if in safe mode
        if safe and len(seq)!=seq_length:
            raise SortSeqError('Sequences are not all the same length.')

        # Fill in array
        for i in range(seq_length-1):
            dichar = seq[i:i+2]
            k = c_to_i_dict[<bytes> dichar]
            mat[n,num_dichars*i+k] = 1
    return mat

def seqs2array_for_pairmodel(list seq_list, bytes seq_type, safe=True):
    """
    Converts a list of sequences (all of which must be the same length) 
    to a numpy array to be used for neighbor model evalution
    """
    cdef np.ndarray[DTYPE_t, ndim=2] mat
    cdef bytes seq, dichar
    cdef int num_seqs, seq_length, num_dichars, i, n, k, index
    cdef dict c_to_i_dict

    # Validate seq_type if in safe mode
    if safe and (not seq_type in qc.seqtypes):
        raise SortSeqError('Invalid seq_type: %s.'%seq_type)

    # Get character dictionary
    c_to_i_dict = qc.char_to_nbr_index_dicts[seq_type]
    num_dichars = len(c_to_i_dict)

    # Initialize matrix
    num_seqs = len(seq_list)
    seq_length = len(seq_list[0])
    mat = np.zeros(\
        [num_seqs,round(sp.misc.comb(seq_length,2))*num_dichars], dtype=DTYPE)

    # Fill matrix row by row
    for n, seq in enumerate(seq_list):

        # Validate sequence composition if in safe mode
        if safe and qc.seqerr_re_dict[seq_type].search(seq):
            raise SortSeqError(\
                'Invalid character found in %s sequence.'%seq_type)

        # Validate sequence length if in safe mode
        if safe and len(seq)!=seq_length:
            raise SortSeqError('Sequences are not all the same length.')
        index = 0 
        # Fill in array
        for i,bp in enumerate(seq):        
            for z in range(i+1,seq_length):
                k = (index*num_dichars + c_to_i_dict[bp + seq[z]])
                index = index + 1
                mat[n,k] = 1
    return mat


