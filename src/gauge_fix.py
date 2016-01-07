import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
 
def get_Lambda_mm(P,C):
    '''
    Matrix-matrix correlation
    '''
    D_n = C*C*(P-1)
    D_m = C*P
    Lambda_mm = np.matrix(np.zeros([D_m,D_m]))
    for c in range(C):
        for d in range(C):
            for p in range(P):
                for q in range(P):
                    k = C*p + c
                    l = C*q + d
                    if p==q and c==d:
                        Lambda_mm[k,l] = 1./C - 1./(C**2)
                    elif p==q:
                        Lambda_mm[k,l] = - 1./(C**2)
                    else:
                        Lambda_mm[k,l] = 0
    return Lambda_mm
 
 
def get_Lambda_nn(P,C):
    '''
    Neighbor-neighbor correlation
    '''
    D_n = C*C*(P-1)
    D_m = C*P
    Lambda_nn = np.matrix(np.zeros([D_n,D_n]))
    for c1 in range(C):
        for c2 in range(C):
            for p in range(P-1):
                for d1 in range(C):
                    for d2 in range(C):
                        for q in range(P-1):
                            k = C*C*p + C*c1 + c2
                            l = C*C*q + C*d1 + d2
                            if p==q:
                                if c1==d1 and c2==d2:
                                    Lambda_nn[k,l] = 1./(C**2) - 1./(C**4)
                                else:
                                    Lambda_nn[k,l] = - 1./(C**4)
                            elif p==q-1:
                                if c2==d1:
                                    Lambda_nn[k,l] = 1./(C**3) - 1./(C**4)
                                else:
                                    Lambda_nn[k,l] = - 1./(C**4)
                            elif p==q+1:
                                if c1==d2:
                                    Lambda_nn[k,l] = 1./(C**3) - 1./(C**4)
                                else:
                                    Lambda_nn[k,l] = - 1./(C**4)
                            else:
                                Lambda_nn[k,l] = 0
    return Lambda_nn
 
 
def get_Lambda_mn(P,C):
    '''
    Matrix-neighbor correlation
    '''
    D_n = C*C*(P-1)
    D_m = C*P
    Lambda_mn = np.matrix(np.zeros([D_m,D_n]))
    for c in range(C):
        for p in range(P):
            for d1 in range(C):
                for d2 in range(C):
                    for q in range(P-1):
                        k = C*p + c
                        l = C*C*q + C*d1 + d2
                        if p==q:
                            if c==d1:
                                Lambda_mn[k,l] = 1./C - 1./(C**3)
                            else:
                                Lambda_mn[k,l] = - 1./(C**3)
                        if p==q+1:
                            if c==d2:
                                Lambda_mn[k,l] = 1./C - 1./(C**3)
                            else:
                                Lambda_mn[k,l] = - 1./(C**3)
                        else:
                            Lambda_mn[k,l] = 0
    return Lambda_mn
 
 
# Neighbor-matrix correlation
def get_Lambda_nm(P,C):
    '''
    Neighbor-matrix correlation
    '''
    Lambda_nm = get_Lambda_mn(P,C).T
    return Lambda_nm
 
 
def fix_matrix(matrix_model, verbose=False, rcond=1.0E-10):
    """
    Transforms a matrix model into the canonical gauge.
     
    Keyword arguments:
        matrix_model - A P x C matrix, P = # positions, C = # characters
        verbose - Prints dimension of matrix and the computed number of gauge freedoms to stdout
        rcond - Relative cutoff for singular values; passed to np.linalg.pinv().
            used to compute the number of gauge freedoms.
     
    Returns:
        fixed_matrix_model - The gauge-fixed matrix
    """
    # Read dimensions of matrix
    P = matrix_model.shape[0]
    C = matrix_model.shape[1]
 
    # Compute projection matrix
    Lambda_mm = get_Lambda_mm(P,C)
    Lambda_mm_pinv, rank = sp.linalg.pinv(Lambda_mm, return_rank=True)
    Proj = np.matrix(Lambda_mm_pinv)*Lambda_mm
     
    # Print number of gauge freedoms if requested
    if verbose:
        print 'C = %d, P = %d'%(C,P)
        print 'Theoretical: dim(G) = P = %d'%P
        print 'Computational: dim(G) = %d'%(D - rank)
 
    # Convert matrix to vector
    matrix_model_vec = np.matrix(matrix_model.flatten()).T
 
    # Project matrix vector
    proj_matrix_model_vec = Proj*matrix_model_vec
 
    # Convert result back to matrix
    fixed_matrix_model = np.array(proj_matrix_model_vec.reshape([P,C]))
     
    # Return matrix model
    return fixed_matrix_model
 
def fix_neighbor(neighbor_model, verbose=False, rcond=1.0E-10):
    """
    Transforms a matrix model into the canonical gauge.
     
    Keyword arguments:
        neighbor_model - A (P-1) x (C^2) matrix, P = # positions, C = # characters
        verbose - Prints dimension of matrix and the computed number of gauge freedoms to stdout
        rcond - Relative cutoff for singular values; passed to np.linalg.pinv().
            used to compute the number of gauge freedoms.
     
    Returns:
        fixed_matrix_model - The gauge-fixed matrix
    """
    # Read dimensions of neighbor model
    P = neighbor_model.shape[0]+1
    Csq = neighbor_model.shape[1]
    C = int(np.sqrt(Csq))
 
    # Compute projection matrix
    Lambda_nn = get_Lambda_nn(P,C)
    Lambda_nn_pinv, rank = \
        sp.linalg.pinv(Lambda_nn, return_rank=True,  rcond=1.0E-10)
    Proj = np.matrix(Lambda_nn_pinv)*Lambda_nn_pinv
     
    # Print number of gauge freedoms if requested
    if verbose:
        print 'C = %d, P = %d'%(C,P)
        print 'Theoretical: dim(G) = (P-1) + (C-1)*(P-2) = %d'%((P-1) + (C-1)*(P-2))
        print 'Computational: Rank = %d => dim(G) = %d'%(rank, D - rank)
     
    # Convert matrix to vector
    neighbor_model_vec = np.matrix(neighbor_model.flatten()).T
 
    # Project matrix vector
    proj_neighbor_model_vec = Proj*neighbor_model_vec
 
    # Convert result back to matrix
    fixed_neighbor_model = np.array(proj_neighbor_model_vec.reshape([P-1,C*C]))
    return fixed_neighbor_model
 
def neighbor2matrix(neighbor_model, verbose=False, rcond=1.0E-10):
    """
    Transforms a matrix model into the canonical gauge.
     
    Keyword arguments:
        neighbor_model - A (P-1) x (C^2) matrix, P = # positions, C = # characters
        verbose - Prints dimension of matrix and the computed number of gauge freedoms to stdout
        rcond - Relative cutoff for singular values; passed to np.linalg.pinv().
            used to compute the number of gauge freedoms.
     
    Returns:
        matrix_model - The matrix projection of the neighbor model
    """
    # Read dimensions of neighbor model
    P = neighbor_model.shape[0]+1
    Csq = neighbor_model.shape[1]
    C = int(np.sqrt(Csq))
     
    # Get correlation matrices
    Lambda_mm = get_Lambda_mm(P,C)
    Lambda_mn = get_Lambda_mn(P,C)
 
    # Compute projection matrix
    Lambda_mm_pinv, rank = sp.linalg.pinv(Lambda_mm, return_rank=True)
    Proj = np.matrix(Lambda_mm_pinv)*Lambda_mn
     
    # Print number of gauge freedoms if requested
    if verbose:
        print 'C = %d, P = %d'%(C,P)
        print 'Theoretical: dim(G_mm) = P = %d'%P
        print 'Computational: dim(G_mm) = %d'%(D - rank)
 
    # Convert matrix to vector
    neighbor_model_vec = np.matrix(neighbor_model.flatten()).T
 
    # Project matrix vector
    proj_matrix_model_vec = Proj*neighbor_model_vec
 
    # Convert result back to matrix
    matrix_model = np.array(proj_matrix_model_vec.reshape([P,C]))
     
    # Return matrix model
    return matrix_model
 
def matrix2neighbor(matrix_model, verbose=False, rcond=1.0E-10):
    """
    Transforms a matrix model into a neighbor model.
     
    Keyword arguments:
        matrix_model - A P x C matrix, P = # positions, C = # characters
        verbose - Prints dimension of matrix and the computed number of gauge freedoms to stdout
        rcond - Relative cutoff for singular values; passed to np.linalg.pinv().
            used to compute the number of gauge freedoms.
     
    Returns:
        neighbor_model - The gauge-fixed matrix
    """
    # Read dimensions of neighbor model
    P = matrix_model.shape[0]
    C = matrix_model.shape[1]
     
    # Get correlation matrices
    Lambda_nn = get_Lambda_nn(P,C)
    Lambda_nm = get_Lambda_nm(P,C)
 
    # Compute projection matrix
    Lambda_nn_pinv, rank = sp.linalg.pinv(Lambda_nn, return_rank=True)
    Proj = np.matrix(Lambda_nn_pinv)*Lambda_nm
     
    # Print number of gauge freedoms if requested
    if verbose:
        print 'C = %d, P = %d'%(C,P)
        print 'Theoretical: dim(G_nn) = (P-1) + (C-1)*(P-2) = %d'%((P-1) + (C-1)*(P-2))
        print 'Computational: Rank = %d => dim(G_nn) = %d'%(rank, D - rank)
         
    # Convert matrix to vector
    matrix_model_vec = np.matrix(matrix_model.flatten()).T
 
    # Project matrix vector
    neighbor_model_vec = Proj*matrix_model_vec
 
    # Convert result back to matrix
    neighbor_model = np.array(neighbor_model_vec.reshape([P-1,C*C]))
     
    # Return matrix model
    return neighbor_model
