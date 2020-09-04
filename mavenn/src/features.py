import numpy as np


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

    # Get additive ohe
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
    assert K == keep.ravel().sum(), f"K={K} doesn't match keep.ravel().sum()={keep.ravel().sum()} "
    if verbose:
        print(f"K = {K} features")

    # Compute ohe for features
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


def _validate_seqs(seqs, alphabet, restrict_seqs_to_alphabet=True):
    """
    Makes sure that seqs is an array of equal-length sequences
    drawn from the set of characters in alphabet. Returns
    a version of seqs cast as a numpy array of strings.
    """

    # Cast as np.array
    if isinstance(seqs, str):
        seqs = np.array([seqs])
    elif isinstance(seqs, list):
        seqs = np.array(seqs).astype(str)
    elif isinstance(seqs, pd.Series):
        seqs = seqs.values.astype(str)
    else:
        assert False, f'type(seqs)={type(seqs)} is invalid.'

    # Make sure array is 1D
    assert len(seqs.shape) == 1, f'seqs should be 1D; seqs.shape={seqs.shape}'

    # Get length and make sure its >= 1
    N = len(seqs)
    assert N >= 1, f'N={N} must be >= 1'

    # Make sure all seqs are the same length
    lengths = np.unique([len(seq) for seq in seqs])
    assert len(
        lengths == 1), f"Sequences should all be the same length; found multiple lengths={lengths}"
    L = lengths[0]

    # Make sure sequences only contain characters in alphabet
    if restrict_seqs_to_alphabet:
        seq_chars = set(''.join(seqs))
        alphabet_chars = set(alphabet)
        assert seq_chars <= alphabet_chars, \
            f"seqs contain the following characters not in alphabet: {seq_chars-alphabet_chars}"

    return seqs


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
    seqs = _validate_seqs(seqs, alphabet, restrict_seqs_to_alphabet)

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
    seqs = _validate_seqs(seqs, alphabet, restrict_seqs_to_alphabet=True)

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
    seqs = _validate_seqs(seqs, alphabet, restrict_seqs_to_alphabet=True)

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

