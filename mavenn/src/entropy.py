"""entropy.py: Utilities to estimate information-theoretic quantities."""
import numpy as np
import pdb

from mavenn.src import _npeet as ee
from mavenn.src.validate import validate_1d_array
from mavenn.src.error_handling import handle_errors, check

@handle_errors
def entropy_continuous(x,
                       knn=5,
                       resolution=.01,
                       uncertainty=True,
                       num_subsamples=25,
                       verbose=False):
    """
    Estimate the entropy of a continuous univariate variable.

    Parameters
    ----------
    x: (array-like of floats)
        Continuous x-values. Must be castable as a
        Nx1 numpy array where N=len(x).

    knn: (int>0)
        Number of nearest neighbors to use in the KSG estimator.

    resolution: (float>0)
        Amount to fuzz up the data, in units of the standard
        deviation of x.

    uncertainty: (bool)
        Whether to estimate the uncertainty of the MI estimate.
        Substantially increases runtime if True.

    num_subsamples: (int > 0)
        Number of subsamples to use if estimating uncertainty.

    verbose: (bool)
        Whether to print results and execution time.

    Returns
    -------
    H: (float)
        Entropy estimate in bits

    dH: (float >= 0)
        Uncertainty estimate in bits. Zero if uncertainty=False is set.
    """
    # TODO: Implement input checks

    # Get number of datapoints
    N = len(x)

    # Reshape to Nx1 array
    x = np.array(x).reshape(N, 1)

    # Fuzz up data
    x_scale = x.std(ddof=1)
    x += resolution * x_scale * np.random.randn(*x.shape)

    # Get best H estimate
    H = ee.entropy(x, k=knn, base=2)

    # If user does not want uncertainty, end here.
    if not uncertainty:
        if verbose:
            print(f'Arguments: knn={knn}, num_subsamples={num_subsamples}')
            print(f'Execution time: {t:.4f} sec')
            print(f'Results: H={H:.4f} bits')

        return H

    # If user does request uncertainty, do computations
    else:
        H_subs = np.zeros(num_subsamples)
        for k in range(num_subsamples):
            N_half = int(np.floor(N / 2))
            ix = np.random.choice(N, size=N_half, replace=False).astype(int)
            x_k = x[ix, :]
            H_subs[k] = ee.entropy(x_k, k=knn, base=2)

        # Estimate dI
        dH = np.std(H_subs, ddof=1) / np.sqrt(2)

        if verbose:
            print(f'Arguments: knn={knn}, num_subsamples={num_subsamples}')
            print(f'Execution time: {t:.4f} sec')
            print(f'Results: H={H:.4f} bits, dH={dH:.4f} bits')

        return H, dH


@handle_errors
def mi_mixed(x,
             y,
             knn=5,
             discrete_var='y',
             uncertainty=True,
             num_subsamples=25,
             verbose=False,
             warning=False):
    """
    Estimate mutual information between a continuous and discrete variable.

    Parameters
    ----------
    x: (array-like)
        Continuous or discrete x-values. Must be castable as a
        Nx1 numpy array where N=len(x).

    y: (array-like)
        Continuous or discrete y-values. Must be the same length
        as x and castable as a Nx1 numpy array.

    knn: (int>0)
        Number of nearest neighbors to use in the KSG estimator.

    discrete_var: (str)
        Which variable is discrete. Must be 'x' or 'y'.

    uncertainty: (bool)
        Whether to estimate the uncertainty of the MI estimate.
        Substantially increases runtime if True.

    num_subsamples: (int > 0)
        Number of subsamples to use if estimating uncertainty.

    verbose: (bool)
        Whether to print results and execution time.

    Returns
    -------
    I: (float)
        Mutual information estimate in bits

    dI: (float >= 0)
        Uncertainty estimate in bits. Zero if uncertainty=False is set.
    """
    # TODO: Input checks

    # Deal with choice of discrete_var
    check(discrete_var in ['x', 'y'],
          f'Invalid value for discrete_var={discrete_var}')
    if discrete_var == 'x':
        return mi_mixed(y, x, discrete_var='y')

    N = len(x)
    assert len(y) == N

    # Make sure x and y are 1D arrays
    x = np.array(x).reshape(N, 1)
    y = np.array(y).reshape(N, 1)

    # Get best I estimate
    I = ee.micd(x, y, k=knn, warning=warning)

    # Compute uncertainty if requested
    if uncertainty:

        # Do subsampling to get I_subs
        I_subs = np.zeros(num_subsamples)
        for k in range(num_subsamples):
            N_half = int(np.floor(N / 2))
            ix = np.random.choice(N, size=N_half, replace=False).astype(int)
            x_k = x[ix, :]
            y_k = y[ix, :]
            I_subs[k] = ee.micd(x_k, y_k, k=knn, warning=False)

        # Estimate dI
        dI = np.std(I_subs, ddof=1) / np.sqrt(2)

    # Otherwise, just set to zero
    else:
        dI = 0.0

    # If verbose, print results:
    if verbose:
        print(f'Arguments: knn={knn}, num_subsamples={num_subsamples}')
        print(f'Execution time: {t:.4f} sec')
        print(f'Results: I={I:.4f} bits, dI={dI:.4f} bits')

    # Return results
    return I, dI

@handle_errors
def mi_continuous(x,
                  y,
                  knn=5,
                  uncertainty=True,
                  num_subsamples=25,
                  use_LNC=False,
                  alpha_LNC=.5,
                  verbose=False):
    """
    Estimate mutual information between two continuous variables.

    Uses the KSG estimator, with optional LNC correction.
    Wrapper for methods in the NPEET package.

    Parameters
    ----------
    x: (array-like of floats)
        Continuous x-values. Must be castable as a
        Nx1 numpy array where N=len(x).

    y: (array-like of floats)
        Continuous y-values. Must be the same length
        as x and castable as a Nx1 numpy array.

    knn: (int>0)
        Number of nearest neighbors to use in the KSG estimator.

    uncertainty: (bool)
        Whether to estimate the uncertainty of the MI estimate.
        Substantially increases runtime if True.

    num_subsamples: (int > 0)
        Number of subsamples to use if estimating uncertainty.

    use_LNC: (bool)
        Whether to compute the Local Nonuniform Correction
        (LNC) using the method of Gao et al., 2015.
        Substantially increases runtime if True.

    alpha_LNC: (float in (0,1))
        Value of alpha to use when computing LNC.
        See Gao et al., 2015 for details.

    verbose: (bool)
        Whether to print results and execution time.

    Returns
    -------
    I: (float)
        Mutual information estimate in bits

    dI: (float >= 0)
        Uncertainty estimate in bits. Zero if uncertainty=False is set.
    """
    # TODO: input checks

    N = len(x)
    assert len(y) == N

    # If not LNC_correction, set LNC_alpha=0
    if not use_LNC:
        alpha_LNC = 0

    # Make sure x and y are 1D arrays
    x = np.array(x).ravel()
    y = np.array(y).ravel()

    # Get best I estimate
    I = ee.mi(x, y, k=knn, alpha=alpha_LNC)

    # Compute uncertainty if requested
    if uncertainty:

        # Do subsampling to get I_subs
        assert num_subsamples >= 2, f'Invalid value for num_subsamples={num_subsamples}'
        I_subs = np.zeros(num_subsamples)
        for k in range(num_subsamples):
            N_half = int(np.floor(N / 2))
            ix = np.random.choice(N, size=N_half, replace=False).astype(int)
            x_k = x[ix]
            y_k = y[ix]
            I_subs[k] = ee.mi(x_k, y_k, k=knn, alpha=alpha_LNC)

        # Estimate dI
        dI = np.std(I_subs, ddof=1) / np.sqrt(2)

    # Otherwise, just set to zero
    else:
        dI = 0.0

    # If verbose, print results:
    if verbose:
        # print(f'Arguments: knn={knn}, num_subsamples={num_subsamples}')
        print(f'Execution time: {t:.4f} sec')
        print(f'Results: I={I:.4f} bits, dI={dI:.4f} bits')

    # Return results
    return I, dI

@handle_errors
def I_intrinsic(y_values,
                dy_values,
                verbose=False):
    """
    Compute instrinsic information in a dataset.

    Parameters
    ----------
    y_values: (array-like of floats)
        y values for which mutual information will be computed.

    dy_values: (array-like of floats)
        Represents errors in the y-values.

    Returns
    -------
    I_y_x: (float)
        Mutual information of y given x.
    dI_y_x: (float)
        Error in the estimated ,mutual information I_y_x.
    """
    #TODO: Input checks

    # useful constants
    e = np.exp(1)
    pi = np.pi

    y_values = validate_1d_array(y_values)
    dy_values = validate_1d_array(dy_values)

    # Compute y and dy values to do the estimation on
    y = y_values / np.std(y_values) + 1E-3 * np.random.randn(*y_values.shape)
    dy = dy_values / np.std(y_values)

    # Compute entropy
    H_y, dH_y = entropy_continuous(y, knn=7)
    if verbose:
        print(f'H[y]   = {H_y:+.4f} +- {dH_y:.4f} bits')

    # Use the formula for Gaussian entropy to compute H[y|x]
    Hs_bits = .5 * np.log2(2 * pi * e * dy ** 2)
    N = len(Hs_bits)
    H_ygx = np.mean(Hs_bits)
    dH_ygx = np.std(Hs_bits, ddof=1) / np.sqrt(N)  # Note that need to specify 1 DOF, not that it matters much.

    if verbose:
        print(f'H[y|x] = {H_ygx:+.4f} +- {dH_ygx:.4f} bits')

    # Finally, compute intrinsic information in experiment
    I_y_x = H_y - H_ygx
    dI_y_x = np.sqrt(dH_y ** 2 + dH_ygx ** 2)
    if verbose:
        print(f'I[y;x] = {I_y_x:+.4f} +- {dI_y_x:.4f} bits')

    return I_y_x, dI_y_x

