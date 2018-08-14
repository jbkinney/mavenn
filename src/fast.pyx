from __future__ import print_function
import numpy as np
cimport numpy as np
from cpython cimport bool

#import qc as qc
from mpathic.src import qc
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
    #print('reverse complement')
    #print(dna_str)
    #print(type(dna_str))

    #print(qc.seqerr_re_dict)
    #print(qc.seqerr_re_dict['dna'])
    #print(type(qc.seqerr_re_dict['dna']))
    #print('here')
    # Validate sequence if in safe mode
    if safe and qc.seqerr_re_dict['dna'].search(dna_str.decode('UTF-8')):
        raise SortSeqError('Invalid character found in DNA sequence.')

    # Compute rc sequence
    #print(rc_dict)
    #print(len(dna_str))


    cdef str test_str = ''
    for index in range(len(dna_str)):
    #for index in range(5):
        #print(dna_str.decode('UTF-8')[index])
        letter = dna_str.decode('UTF-8')[index]
        #print(letter)
        #print(rc_dict[letter])
        test_str +=str(rc_dict[letter])

    #print(test_str)

    #cdef bytes c_str = str.encode(test_str)
    cdef bytes c_str = test_str.encode()
    #print(type(test_str))
    #print(type(c_str))
    #print(test_str[::-1])
    #print(c_str)

    cdef bytes B
    #cdef bytes c_str = bytes(b''.join([rc_dict[B] for B in dna_str]))
    return c_str[::-1]


def seq2sitelist(bytes seq, int site_length, safe=True, rc=False):
    """
    Chops a sequence into sites of length site_length
    """
    cdef int seq_length = len(seq)
    cdef int num_sites = seq_length - site_length + 1
    cdef int i
    cdef bytes rcseq

    #print(seq.decode('UTF-8'))

    # Make sure there are actually sites (if in safe mode)
    if safe and num_sites<=0:
        raise SortSeqError('site_length > seq_length.')

    # Initialize site_list
    cdef list site_list = [b'.'*site_length]*num_sites

    # Fill site_list
    if rc:
        #print('seq2sitelist')
        #print(seq)
        #print(type(seq))
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

    #print(seq_type.decode('UTF-8'))
    #print(seq_list)

    print('seq2array_for_mat_model')


    # Validate seq_type if in safe mode
    #if safe and (not seq_type in qc.seqtypes):
    if safe and (not seq_type.decode('UTF-8') in qc.seqtypes):
        raise SortSeqError('Invalid seq_type: %s.'%seq_type)

    # Get character dictionary
    #c_to_i_dict = qc.char_to_mat_index_dicts[seq_type]
    c_to_i_dict = qc.char_to_mat_index_dicts[seq_type.decode('UTF-8')]
    num_chars = len(c_to_i_dict)

    # Initialize matrix
    num_seqs = len(seq_list)
    seq_length = len(seq_list[0])
    #print(seq_length)
    mat = np.zeros([num_seqs,num_chars*seq_length], dtype=DTYPE)

    # Fill matrix row by row
    for n, seq in enumerate(seq_list):

        # Validate sequence composition if in safe mode
        #if safe and qc.seqerr_re_dict[seq_type].search(seq):
        #if safe and qc.seqerr_re_dict[seq_type.decode('UTF-8')].search(seq):

        #print(seq_type)
        #print(seq_type.decode('UTF-8'))
        #print(seq.hex())
        #print(seq)
        #print(seq.decode('UTF-8'))
        #print(qc.seqerr_re_dict)
        #print(qc.seqerr_re_dict['dna'])

        #if safe and qc.seqerr_re_dict[seq_type].search(seq.decode('UTF-8')):
        if safe and qc.seqerr_re_dict[seq_type.decode('UTF-8')].search(seq.decode('UTF-8')):
            raise SortSeqError(\
                'Invalid character found in %s sequence.'%seq_type.decode('UTF-8'))

        # Validate sequence length if in safe mode
        if safe and len(seq)!=seq_length:
            raise SortSeqError('Sequences are not all the same length.')

        #print(c_to_i_dict)
        #print(c_to_i_dict['A'])
        #print(type(c_to_i_dict['A']))

        # Fill in array
        for i, c in enumerate(seq):
            k = c_to_i_dict[c.decode('UTF-8')]
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

    #if safe and (not seq_type in qc.seqtypes):
    if safe and (not seq_type.decode('UTF-8') in qc.seqtypes):
        raise SortSeqError('Invalid seq_type: %s.'%seq_type)

    # Get character dictionary
    #c_to_i_dict = qc.char_to_nbr_index_dicts[seq_type]
    c_to_i_dict = qc.char_to_nbr_index_dicts[seq_type.decode('UTF-8')]
    num_dichars = len(c_to_i_dict)

    # Initialize matrix
    num_seqs = len(seq_list)
    seq_length = len(seq_list[0])
    mat = np.zeros([num_seqs,num_dichars*(seq_length-1)], dtype=DTYPE)

    # Fill matrix row by row
    for n, seq in enumerate(seq_list):

        # Validate sequence composition if in safe mode
        #if safe and qc.seqerr_re_dict[seq_type].search(seq):
        if safe and qc.seqerr_re_dict[seq_type.decode('UTF-8')].search(seq.decode('UTF-8')):
            raise SortSeqError(\
                'Invalid character found in %s sequence.'%seq_type.decode('UTF-8'))

        # Validate sequence length if in safe mode
        if safe and len(seq)!=seq_length:
            raise SortSeqError('Sequences are not all the same length.')

        # Fill in array
        for i in range(seq_length-1):
            dichar = seq[i:i+2]
            k = c_to_i_dict[<bytes> dichar.decode('UTF-8')]
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


