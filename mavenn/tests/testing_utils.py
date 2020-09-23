# Standard imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb

# Initialize global counts
global_success_counter = 0
global_fail_counter = 0

# Common success and fail lists
bool_fail_list = [0, -1, 'True', 'x', 1]
bool_success_list = [False, True]

# MAVE-NN imports
from mavenn.src.error_handling import check, handle_errors
from mavenn.src.validate import validate_alphabet, validate_seqs


# helper method for functional test_for_mistake
def test_for_mistake(func, *args, **kw):
    """
    Run a function with the specified parameters and register whether
    success or failure was a mistake

    parameters
    ----------

    func: (function or class constructor)
        An executable function to which *args and **kwargs are passed.

    return
    ------

    None.
    """

    global global_fail_counter
    global global_success_counter

    # print test number
    test_num = global_fail_counter + global_success_counter
    print('Test # %d: ' % test_num, end='')

    # Run function
    obj = func(*args, **kw)
    # Increment appropriate counter
    if obj.mistake:
        global_fail_counter += 1
    else:
        global_success_counter += 1


def test_parameter_values(func,
                          var_name=None,
                          fail_list=[],
                          success_list=[],
                          **kwargs):
    """
    Tests predictable success & failure of different values for a
    specified parameter when passed to a specified function

    parameters
    ----------

    func: (function)
        Executable to test. Can be function or class constructor.

    var_name: (str)
        Name of variable to test. If not specified, function is
        tested for success in the absence of any passed parameters.

    fail_list: (list)
        List of values for specified variable that should fail

    success_list: (list)
        List of values for specified variable that should succeed

    **kwargs:
        Other keyword variables to pass onto func.

    return
    ------

    None.

    """

    # If variable name is specified, test each value in fail_list
    # and success_list
    if var_name is not None:

        # User feedback
        print("Testing %s() parameter %s ..." % (func.__name__, var_name))

        # Test parameter values that should fail
        for x in fail_list:
            kwargs[var_name] = x
            test_for_mistake(func=func, should_fail=True, **kwargs)

        # Test parameter values that should succeed
        for x in success_list:
            kwargs[var_name] = x
            test_for_mistake(func=func, should_fail=False, **kwargs)

        print("Tests passed: %d. Tests failed: %d.\n" %
              (global_success_counter, global_fail_counter))

    # Otherwise, make sure function without parameters succeeds
    else:

        # User feedback
        print("Testing %s() without parameters." % func.__name__)

        # Test function
        test_for_mistake(func=func, should_fail=False, **kwargs)

    # close all figures that might have been generated
    plt.close('all')



@handle_errors
def additive_model_features(seqs, alphabet, restrict_seqs_to_alphabet=True):
    """
    Compute additive model features from a list of sequences.
    For sequences of length L and an alphabet of length C,
        K = 1 + L*C
    features are encoded, representing constant and
    additive features.

    parameters
    ----------

    seqs: (str or array of str)
        Array of N sequences to encode.

    alphabet: (array of characters)
        Array of C characters from which to build features

    restrict_seqs_to_alphabet: (bool)
        Whether to throw an error if seqs contains characters
        not in alphabet. If False, characters in seqs
        that are not in alphabet will have feature value 0 for
        features that reference that character's position. This
        might cause problems to arise during gauge fixing.

    returns
    -------

    x: (2D np.ndarray)
        A binary numpy array of shape (N,K)

    names: (list of str)
        A list of feature names
    """

    # Validate seqs
    seqs = validate_seqs(seqs, alphabet, restrict_seqs_to_alphabet)

    # Get constant features
    N = len(seqs)
    x_0 = np.ones(N).reshape(N, 1)
    features_0 = ['x_0']

    # Get additive features
    x_lc, features_lc = _seqs_to_x_lc(seqs, alphabet, return_features=True)

    # Concatenate features and feature names
    x = np.hstack([x_0, x_lc])
    names = features_0 + features_lc
    return x, names

@handle_errors
def _seqs_to_x_lc(seqs, alphabet,
                  return_features=False,
                  verbose=False,
                  seq_to_print=0,
                  features_to_print=20):
    # Get N, L, C
    N = len(seqs)
    L = len(seqs[0])
    C = len(alphabet)

    # Get vectors of unique lengths and characters
    l_uniq = np.arange(L).astype(int)
    c_uniq = np.array(list(alphabet))

    # Get (N,L) matrix of sequence characters
    seq_mat = np.array([list(seq) for seq in seqs])

    # Create (L,C) grids of positions and characters
    l_add_grid = np.tile(np.reshape(l_uniq, [L, 1]), [1, C])
    c_add_grid = np.tile(np.reshape(c_uniq, [1, C]), [L, 1])

    # Create (N,L,C) grid of characters in sequences
    seq_add_grid = np.tile(np.reshape(seq_mat, [N, L, 1]), [1, 1, C])

    # Compute (N,L,C) grid of one-hot encoded values
    x_add_grid = (seq_add_grid == c_add_grid[np.newaxis, :, :])

    # Compute number of features K
    K = L * C

    # Compute flattened lists positions and characters
    l_add = l_add_grid.reshape(K)
    c_add = c_add_grid.reshape(K)

    # Create one-hot encoded matrix to return
    x_add = x_add_grid.reshape(N, K)

    # Print features if requested
    if verbose:
        n = seq_to_print
        print(f'x[{n}] = {seqs[n]}')
        ix = x_add[n, :]
        cs = c_add[ix]
        ls = l_add[ix]
        k_max = min(ix.sum(), features_to_print)
        for k in range(k_max):
            name = f"x[{n}]_{ls[k]}:{cs[k]} = True"
            print(name)

    # If return features, create list of feature names and return with x_add
    if return_features:
        feature_names = [f'x_{l_add[k]}:{c_add[k]}' for k in range(K)]
        return x_add, feature_names

    # Otherwise, just return x_add
    else:
        return x_add


@handle_errors
def _seqs_to_x_lclc(seqs, alphabet,
                    return_features=False,
                    verbose=False,
                    seq_to_print=0,
                    features_to_print=20,
                    feature_mask='pairwise'):
    # Get N, L, C
    N = len(seqs)
    L = len(seqs[0])
    C = len(alphabet)

    # Get vectors of unique lengths and characters
    l_uniq = np.arange(L).astype(int)
    c_uniq = np.array(list(alphabet))

    # Get (N,L) matrix of sequence characters
    seq_mat = np.array([list(seq) for seq in seqs])

    # Get additive x_ohe
    x_add = _seqs_to_x_lc(seqs, alphabet)

    # Create (L,C) grids of positions and characters
    l1_grid = np.tile(np.reshape(l_uniq, [L, 1, 1, 1]), [1, C, L, C])
    c1_grid = np.tile(np.reshape(c_uniq, [1, C, 1, 1]), [L, 1, L, C])
    l2_grid = np.tile(np.reshape(l_uniq, [1, 1, L, 1]), [L, C, 1, C])
    c2_grid = np.tile(np.reshape(c_uniq, [1, 1, 1, C]), [L, C, L, 1])

    # Get indices for collapsing dimensions
    if feature_mask == 'neighbor':
        keep = (l1_grid == l2_grid - 1)
        K = int((C ** 2) * (L - 1))

    elif feature_mask == 'pairwise':
        keep = (l1_grid < l2_grid)
        K = int((C ** 2) * L * (L - 1) / 2)

    else:
        print(f'Invalid feature_mask={feature_mask}')

    check(K == keep.ravel().sum(),
          f"This shouldnt ever happen." 
          f"K={K} doesn't match keep.ravel().sum()={keep.ravel().sum()}")

    if verbose:
        print(f"K = {K} features")

    # Compute x_ohe for features
    x_add1 = x_add.reshape(N, L, C, 1, 1)
    x_add2 = x_add.reshape(N, 1, 1, L, C)
    x_pair = (x_add1 * x_add2)[:, keep]

    # Print parameters
    l1_pair = l1_grid[keep]
    l2_pair = l2_grid[keep]
    c1_pair = c1_grid[keep]
    c2_pair = c2_grid[keep]

    # Print features if requested
    if verbose:
        n = seq_to_print
        print(f'x[{n}] = {seqs[n]}')
        ix = x_pair[n, :]
        c1s = c1_pair[ix]
        l1s = l1_pair[ix]
        c2s = c2_pair[ix]
        l2s = l2_pair[ix]
        k_max = min(ix.sum(), features_to_print)
        for k in range(k_max):
            name = f"x[{n}]_{l1s[k]}:{c1s[k]},{l2s[k]}:{c2s[k]} = True"
            print(name)

    # If return_features, create a list of feature names and return with x_pair
    if return_features:
        feature_names = [
            f'x_{l1_pair[k]}:{c1_pair[k]},{l2_pair[k]}:{c2_pair[k]}' for k in
            range(K)]
        return x_pair, feature_names
    # Otherwise, just return x_pair
    else:
        return x_pair

@handle_errors
def neighbor_model_features(seqs, alphabet, restrict_seqs_to_alphabet=True):
    """
    Compute neighbor model features from a list of sequences.
    For sequences of length L and an alphabet of length C,
        K = 1 + L*C + (L-1)*C*C
    features are encoded, representing constant, additive,
    and neighbor features.

    parameters
    ----------

    seqs: (str or array of str)
        Array of N sequences to encode.

    alphabet: (array of characters)
        Array of C characters from which to build features

    restrict_seqs_to_alphabet: (bool)
        Whether to throw an error if seqs contains characters
        not in alphabet. If False, characters in seqs
        that are not in alphabet will have feature value 0 for
        features that reference that character's position. This
        might cause problems to arise during gauge fixing.

    returns
    -------

    x: (2D np.ndarray)
        A binary numpy array of shape (N,K)

    names: (list of str)
        A list of feature names
    """

    # Validate seqs
    seqs = validate_seqs(seqs, alphabet, restrict_seqs_to_alphabet)

    # Get constant features
    N = len(seqs)
    x_0 = np.ones(N).reshape(N, 1)
    features_0 = ['x_0']

    # Get additive features
    x_lc, features_lc = _seqs_to_x_lc(seqs, alphabet, return_features=True)

    # Get additive features
    x_lclc, features_lclc = _seqs_to_x_lclc(seqs, alphabet,
                                            return_features=True,
                                            feature_mask="neighbor")

    # Concatenate features
    x = np.hstack([x_0, x_lc, x_lclc])
    names = features_0 + features_lc + features_lclc
    return x, names

@handle_errors
def pairwise_model_features(seqs, alphabet, restrict_seqs_to_alphabet=True):
    """
    Compute pairwise model features from a list of sequences.
    For sequences of length L and an alphabet of length C,
        K = 1 + L*C + (L*(L-1)/2)*C*C
    features are encoded, representing constant, additive,
    and unique pairwise features.

    parameters
    ----------

    seqs: (str or array of str)
        Array of N sequences to encode.

    alphabet: (array of characters)
        Array of C characters from which to build features

    restrict_seqs_to_alphabet: (bool)
        Whether to throw an error if seqs contains characters
        not in alphabet. If False, characters in seqs
        that are not in alphabet will have feature value 0 for
        features that reference that character's position. This
        might cause problems to arise during gauge fixing.

    returns
    -------

    x: (2D np.ndarray)
        A binary numpy array of shape (N,K)

    names: (list of str)
        A list of feature names
    """
    # Validate seqs
    seqs = validate_seqs(seqs, alphabet, restrict_seqs_to_alphabet=True)

    # Get constant features
    N = len(seqs)
    x_0 = np.ones(N).reshape(N, 1)
    features_0 = ['x_0']

    # Get additive features
    x_lc, features_lc = _seqs_to_x_lc(seqs, alphabet, return_features=True)

    # Get additive features
    x_lclc, features_lclc = _seqs_to_x_lclc(seqs, alphabet,
                                            return_features=True,
                                            feature_mask="pairwise")

    # Concatenate features
    x = np.hstack([x_0, x_lc, x_lclc])
    names = features_0 + features_lc + features_lclc
    return x, names