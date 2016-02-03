import numpy as np
cimport numpy as np

DTYPE = np.int
ctypedef np.int_t DTYPE_t

# Convert a sequence of length L of ACGT to a number
dnabase_to_index_dict = {
'A':0,
'C':1,
'G':2,
'T':3}


# Cython funciton
def dna_list2mat_x(list seq_list):
    cdef int N = len(seq_list)
    cdef int L = len(seq_list[0])
    cdef np.ndarray[DTYPE_t, ndim=2] a = np.zeros([N,4*L], dtype=DTYPE)
    cdef bytes seq
    cdef int i, n, b
    cdef char c
    for n, seq in enumerate(seq_list):
        assert len(seq)==L
        for i, c in enumerate(seq):
            b = dnabase_to_index_dict[<bytes>c]
            a[n,4*i+b] = 1
    return a
    
# Regular python
def dna_list2mat(seq_list):
    N = len(seq_list)
    L = len(seq_list[0])
    a = np.zeros([N,4*L], dtype=int)
    for n, seq in enumerate(seq_list):
        assert len(seq)==L
        for i, c in enumerate(seq):
            b = dnabase_to_index_dict[c]
            a[n,4*i+b] = 1
    return a


