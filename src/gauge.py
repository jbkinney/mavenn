import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats

class Info:
    pass;

# Defines different types of sequences that can be processed
alphabets = {
    'DNA':'ACGT', 
    'RNA':'ACGU',
    'protein':'*ACDEFGHIKLMNPQRSTVWY'
}

def load_matrix(filename, wtseq, mutrate, seqtype='DNA', normalize=False, \
    features=False, verbose=False, rcond=1E-10):
    '''
    Loads a matrix model from specified file
    '''
    df_matrix = pd.read_csv(filename,delim_whitespace=True)
    del df_matrix ['pos']
    matrix_model = np.array(df_matrix)
    matrix_cg = fix_matrix(matrix_model, wtseq=wtseq, mutrate=mutrate, \
        seqtype=seqtype, normalize=normalize, verbose=verbose, rcond=rcond)
    if features:
        matrix_features = [v.split('_')[-1] for v in df_matrix.columns.values]
        return matrix_cg, matrix_features

    else:
        return matrix_cg


def load_neighbor(filename, wtseq, mutrate, seqtype='DNA', normalize=False, \
    features=False, verbose=False, rcond=1E-10):
    '''
    Loads a neighbor model from specified file
    '''
    df_neighbor = pd.read_csv(filename,delim_whitespace=True)
    del df_neighbor ['pos']
    neighbor_model = np.array(df_neighbor)
    neighbor_cg = fix_neighbor(neighbor_model, wtseq=wtseq, mutrate=mutrate, \
        seqtype=seqtype, normalize=normalize, verbose=verbose, rcond=rcond)
    if features:
        neighbor_features = [v.split('_')[-1] for v in \
                             df_neighbor.columns.values]
        return neighbor_cg, neighbor_features

    else:
        return neighbor_cg


def get_Lambda_mm(wtseq,mutrate,seqtype='DNA'):
    '''
    Matrix model sequence covariance matrix
    '''
    # Get length of sequence
    P = len(wtseq)

    # Get alphabet
    alphabet = alphabets[seqtype]
    C = len(alphabet)

    # Convert string seq to nseq; throws exeception if letter isn't found
    nseq = [alphabet.index(letter) for letter in wtseq]

    # Get mutation rate
    mutrate = min(mutrate, (C-1.)/C)

    # Build up lambda matrix
    D_m = C*P
    Lambda_mm = np.matrix(np.zeros([D_m,D_m]))
    for p in range(P):
        for q in range(P):
            for c in range(C):
                for d in range(C):

                    # Compute prob of base at the two positions
                    prob_p = 1-mutrate if nseq[p]==c else (mutrate/(C-1))
                    prob_q = 1-mutrate if nseq[q]==d else (mutrate/(C-1))

                    # Fill in element of Lambda
                    k = C*p + c
                    l = C*q + d
                    if p==q and c==d:
                        Lambda_mm[k,l] = prob_p - prob_p*prob_q
                    elif p==q:
                        Lambda_mm[k,l] = - prob_p*prob_q
                    else:
                        Lambda_mm[k,l] = 0
    return Lambda_mm


def get_Lambda_nn(wtseq,mutrate,seqtype='DNA'):
    '''
    Neighbor model sequence covariance matrix
    '''

    # Get length of sequence
    P = len(wtseq)

    # Get alphabet
    alphabet = alphabets[seqtype]
    C = len(alphabet)

    # Convert string seq to nseq; throws exeception if letter isn't found
    nseq = [alphabet.index(letter) for letter in wtseq]

    # Get mutation rate
    mutrate = min(mutrate, (C-1.)/C)
    assert mutrate < 1, 'mutation rate of %f is too high.'%mutrate

    # Create matrix of probs at each position
    probs = np.zeros([P,C])
    for p in range(P):
        for c in range(C):
            if nseq[p]==c:
                probs[p,c] = 1.-mutrate
            else:
                probs[p,c] = mutrate/(C-1.)

    # Create sequence covariance matrix
    D_n = C*C*(P-1)
    Lambda_nn = np.matrix(np.zeros([D_n,D_n]))
    for c1 in range(C):
        for c2 in range(C):
            for p in range(P-1):
                for d1 in range(C):
                    for d2 in range(C):
                        for q in range(P-1):

                            # Extract relevant probabilities
                            u = probs[p,c1]
                            v = probs[p+1,c2]
                            x = probs[q,d1]
                            y = probs[q+1,d2]

                            # Compute prob of both features
                            if p==q:
                                if c1==d1 and c2==d2:
                                    prob_pq = u*v 
                                else:
                                    prob_pq = 0
                            elif p==q-1:
                                if c2==d1:
                                    prob_pq = u*v*y
                                else:
                                    prob_pq = 0
                            elif p==q+1:
                                if c1==d2:
                                    prob_pq = x*u*v
                                else:
                                    prob_pq = 0
                            else:
                                prob_pq = u*v*x*y

                            # Compute marginal probs
                            prob_p = u*v
                            prob_q = x*y

                            # Get linear coordinates of p and q
                            k = C*C*p + C*c1 + c2
                            l = C*C*q + C*d1 + d2

                            # Record covariance
                            Lambda_nn[k,l] = prob_pq - prob_p*prob_q
    return Lambda_nn


def get_Lambda_mn(wtseq,mutrate, seqtype='DNA'):
    '''
    Neighbor model sequence covariance matrix
    '''

    # Get length of sequence
    P = len(wtseq)

    # Get alphabet
    alphabet = alphabets[seqtype]
    C = len(alphabet)

    # Convert string seq to nseq; throws exeception if letter isn't found
    nseq = [alphabet.index(letter) for letter in wtseq]

    # Get mutation rate
    mutrate = min(mutrate, (C-1.)/C)
    assert mutrate < 1, 'mutation rate of %f is too high.'%mutrate

    # Create matrix of probs at each position
    probs = np.zeros([P,C])
    for p in range(P):
        for c in range(C):
            if nseq[p]==c:
                probs[p,c] = 1.-mutrate
            else:
                probs[p,c] = mutrate/(C-1.)

    # Create sequence covariance matrix
    D_n = C*C*(P-1)
    D_m = C*P
    Lambda_mn = np.matrix(np.zeros([D_m,D_n]))
    for c in range(C):
        for p in range(P):
            for d1 in range(C):
                for d2 in range(C):
                    for q in range(P-1):

                            # Extract relevant probabilities
                            u = probs[p,c]
                            x = probs[q,d1]
                            y = probs[q+1,d2]

                            # Compute prob of both features
                            if p==q:
                                if c==d1:
                                    prob_pq = u*y 
                                else:
                                    prob_pq = 0
                            elif p==q+1:
                                if c==d2:
                                    prob_pq = x*u
                                else:
                                    prob_pq = 0
                            else:
                                prob_pq = u*x*y

                            # Compute marginal probs
                            prob_p = u
                            prob_q = x*y

                            # Get linear coordinates of p and q
                            k = C*p + c
                            l = C*C*q + C*d1 + d2

                            # Record covariance
                            Lambda_mn[k,l] = prob_pq - prob_p*prob_q
    return Lambda_mn


def get_Lambda_nm(wtseq, mutrate, seqtype='DNA'):
    '''
    Neighbor-matrix covariance matrix
    '''
    Lambda_nm = get_Lambda_mn(wtseq, mutrate, seqtype).T
    return Lambda_nm


def fix_matrix(matrix_model, wtseq='', mutrate=1, seqtype='DNA', \
    verbose=False, rcond=1.0E-10, normalize=False):
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
    D = P*C

    # Create wt seq if none specified
    if wtseq=='':
        wtseq = 'A'*P

    # Set mutrate if not specified
    if mutrate==1:
        mutrate = 1.0-1.0/C

    # Center matrix
    matrix_model = matrix_model - np.mean(matrix_model,axis=1)[:,np.newaxis]

    # Make sure wtseq is same length as matrix
    assert len(wtseq) == P, 'len(wtseq)==%d, P==%d'%(len(wtseq),P)

    # Compute projection matrix
    Lambda_mm = get_Lambda_mm(wtseq, mutrate, seqtype)

    # Compute projection matrix using SVD
    U, s, Vh = sp.linalg.svd(Lambda_mm)
    tol = s.max()*max(Lambda_mm.shape)*rcond # This seems to be failing!
    rank = sum(s > tol)
    Q_cropped = np.matrix(U[:,s > tol])
    Proj = Q_cropped * Q_cropped.T
    
    # Print number of gauge freedoms if requested
    if verbose:
        print('C = %d, P = %d'%(C,P))
        print('Theoretical: dim(G) = P = %d'%P)
        print('Computational: dim(G) = %d'%(D - rank))

    # Verify number of gauge freedoms
    assert (D - rank) == P # This is throwing errors! Maybe remove. 

    # Convert matrix to vector
    matrix_model_vec = np.matrix(matrix_model.flatten()).T

    # Project matrix vector
    proj_matrix_model_vec = Proj*matrix_model_vec

    # Convert result back to matrix
    fixed_matrix_model = np.array(proj_matrix_model_vec.reshape([P,C]))
    
    # Normalize magnitude if requested
    if normalize:
        matrix_vec = np.matrix(fixed_matrix_model.flatten()).T
        power = (matrix_vec.T*Lambda_mm*matrix_vec)[0,0]
        fixed_matrix_model /= np.sqrt(power)

    # Return matrix model
    return fixed_matrix_model


def fix_neighbor(neighbor_model, wtseq='', mutrate=1, seqtype='DNA', \
    verbose=False, rcond=1.0E-10, normalize=False):
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
    D = (P-1)*C**2

    # Create wt seq if none specified
    if wtseq=='':
        wtseq = 'A'*P

    # Set mutrate if not specified
    if mutrate==1:
        mutrate = 1.0-1.0/C

    # Center matrix
    neighbor_model = neighbor_model \
        - np.mean(neighbor_model,axis=1)[:,np.newaxis]

    # Make sure wtseq is same length as matrix
    assert len(wtseq) == P, 'len(wtseq)==%d, P==%d'%(len(wtseq),P)

    # Compute projection matrix
    Lambda_nn = get_Lambda_nn(wtseq, mutrate, seqtype)

    # Compute projection matrix using SVD
    U, s, Vh = sp.linalg.svd(Lambda_nn)
    tol = s.max()*max(Lambda_nn.shape)*rcond
    rank = sum(s > tol)
    Q_cropped = np.matrix(U[:,s > tol])
    Proj = Q_cropped * Q_cropped.T

    # Print number of gauge freedoms if requested
    if verbose:
        print('C = %d, P = %d'%(C,P))
        print('Theoretical: dim(G) = (P-1) + (C-1)*(P-2) = %d'%((P-1) + (C-1)*(P-2)))
        print('Computational: Rank = %d => dim(G) = %d'%(rank, D - rank))

    # Verify number of gauge freedoms
    assert (D - rank) == ((P-1) + (C-1)*(P-2))
    
    # Convert matrix to vector
    neighbor_model_vec = np.matrix(neighbor_model.flatten()).T

    # Project matrix vector
    proj_neighbor_model_vec = Proj*neighbor_model_vec

    # Convert result back to matrix
    fixed_neighbor_model = np.array(proj_neighbor_model_vec.reshape([P-1,C*C]))
    
    # Normalize magnitude if requested
    if normalize:
        neighbor_vec = np.matrix(fixed_neighbor_model.flatten()).T
        power = (neighbor_vec.T*Lambda_nn*neighbor_vec)[0,0]
        fixed_neighbor_model /= np.sqrt(power)

    return fixed_neighbor_model


def neighbor2matrix(neighbor_model, wtseq, mutrate, seqtype='DNA', \
    verbose=False, rcond=1.0E-10):
    """
    Transforms a matrix model into the canonical gauge using 
    nonuniform library.
    
    Keyword arguments:
        neighbor_model - A (P-1) x (C^2) matrix, P = # positions, C = # characters
        wtseq - wild type sequence about which library seqs are mutated
        mutrate - library mutation reate
        seqtype - type of sequence
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
    D = P*C
    
    # Make sure that wtseq is the right length and
    # seqtype has the right number of characters
    assert P == len(wtseq), "Size conflict: wtseq==%s is length %d; matrix has %d rows"%(wtseq,len(wtseq),P)
    assert C == len(alphabets[seqtype]), "Size conflict: seqtype==%s specifies %d characters; matrix has %d columns"%(seqtype,len(alphabets[seqtype]),C)

    # Get correlation matrices
    Lambda_mm = get_Lambda_mm(wtseq,mutrate,seqtype=seqtype)
    Lambda_mn = get_Lambda_mn(wtseq,mutrate,seqtype=seqtype)

    # Compute projection matrix
    Lambda_mm_pinv, rank = sp.linalg.pinv(Lambda_mm, \
        return_rank=True, rcond=rcond)
    Proj = np.matrix(Lambda_mm_pinv)*Lambda_mn
    
    # Print number of gauge freedoms if requested
    if verbose:
        print('C = %d, P = %d'%(C,P))
        print('Theoretical: dim(G_mm) = P = %d'%P)
        print('Computational: dim(G_mm) = %d'%(D - rank))

    # Verify number of gauge freedoms
    assert (D - rank) == P

    # Convert matrix to vector
    neighbor_model_vec = np.matrix(neighbor_model.flatten()).T

    # Project matrix vector
    proj_matrix_model_vec = Proj*neighbor_model_vec

    # Convert result back to matrix
    matrix_model = np.array(proj_matrix_model_vec.reshape([P,C]))
    
    # Return matrix model
    return matrix_model


def matrix2neighbor(matrix_model, wtseq, mutrate, seqtype='DNA', \
    verbose=False, rcond=1.0E-10):
    """
    Transforms a matrix model into a neighbor model using a nonuniform library.
    
    Keyword arguments:
        matrix_model - A P x C matrix, P = # positions, C = # characters
        wtseq - wild type sequence about which library seqs are mutated
        mutrate - library mutation reate
        seqtype - type of sequence
        verbose - Prints dimension of matrix and the computed number of gauge freedoms to stdout
        rcond - Relative cutoff for singular values; passed to np.linalg.pinv().
            used to compute the number of gauge freedoms.
    
    Returns:
        neighbor_model - The gauge-fixed matrix
    """
    # Read dimensions of neighbor model
    P = matrix_model.shape[0]
    C = matrix_model.shape[1]
    D = (P-1)*C**2
    
    # Make sure that wtseq is the right length and
    # seqtype has the right number of characters
    assert P == len(wtseq), \
        "Size conflict: wtseq==%s is length %d; matrix has %d rows"%(wtseq,len(wtseq),P)
    assert C == len(alphabets[seqtype]), \
        "Size conflict: seqtype==%s specifies %d characters; matrix has %d columns"%(seqtype,len(alphabets[seqtype]),C)

    # Get correlation matrices
    Lambda_nn = get_Lambda_nn(wtseq,mutrate,seqtype=seqtype)
    Lambda_nm = get_Lambda_nm(wtseq,mutrate,seqtype=seqtype)

    # Compute projection matrix
    Lambda_nn_pinv, rank = sp.linalg.pinv(Lambda_nn, \
        return_rank=True, rcond=rcond)
    Proj = np.matrix(Lambda_nn_pinv)*Lambda_nm
    
    # Print number of gauge freedoms if requested
    if verbose:
        print('C = %d, P = %d'%(C,P))
        print('Theoretical: dim(G_nn) = (P-1) + (C-1)*(P-2) = %d'%((P-1) + (C-1)*(P-2)))
        print('Computational: Rank = %d => dim(G_nn) = %d'%(rank, D - rank))
    
    # Verify number of gauge freedoms
    assert (D - rank) == ((P-1) + (C-1)*(P-2))

    # Convert matrix to vector
    matrix_model_vec = np.matrix(matrix_model.flatten()).T

    # Project matrix vector
    neighbor_model_vec = Proj*matrix_model_vec

    # Convert result back to matrix
    neighbor_model = np.array(neighbor_model_vec.reshape([P-1,C*C]))
    
    # Return matrix model
    return neighbor_model


def hierarchical_gauge(neighbor_model, wtseq, mutrate, seqtype='DNA', \
    verbose=False, rcond=1.0E-10, power=False):

    # Fix neighbor in canonical gauge
    neighbor_cg = fix_neighbor(neighbor_model, wtseq=wtseq, \
        mutrate=mutrate, rcond=rcond)

    # Project neighbor to matrix
    matrix_hg = neighbor2matrix(neighbor_cg, wtseq=wtseq, \
        mutrate=mutrate, rcond=rcond)

    # Project matrix back to neighbor
    neighbor_hg_parallel = matrix2neighbor(matrix_hg, \
        wtseq=wtseq, mutrate=mutrate, rcond=rcond)

    # Get perpendicular component of neighbor
    neighbor_hg_perp = neighbor_cg - neighbor_hg_parallel

    if power:
        # Compute power in parallel and perpendicular components, 
        # as well as cross-power, as fraction of total power
        ncg_vec = np.matrix(neighbor_cg.flatten()).T
        mhg_vec = np.matrix(matrix_hg.flatten()).T
        nhg_parallel_vec = np.matrix(neighbor_hg_parallel.flatten()).T
        nhg_perp_vec = np.matrix(neighbor_hg_perp.flatten()).T

        # Get Lambda matrices for computing innner products
        Lambda_mm = get_Lambda_mm(wtseq, mutrate)
        Lambda_nn = get_Lambda_nn(wtseq, mutrate)

        # Compute power; store in "info" class
        o = Info()
        o.mutrate = mutrate
        o.wtseq = wtseq
        o.total_power = (ncg_vec.T*Lambda_nn*ncg_vec)[0,0]
        o.matrix_power = (mhg_vec.T*Lambda_mm*mhg_vec)[0,0]
        o.parallel_power = (nhg_parallel_vec.T*Lambda_nn*nhg_parallel_vec)[0,0]
        o.perp_power = (nhg_perp_vec.T*Lambda_nn*nhg_perp_vec)[0,0]
        o.cross_power = (nhg_parallel_vec.T*Lambda_nn*nhg_perp_vec)[0,0]

        # Return fixed model
        return matrix_hg, neighbor_hg_perp, o

    else:
        # Return fixed model
        return matrix_hg, neighbor_hg_perp
