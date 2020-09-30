# Standard imports
import numpy as np
import pandas as pd
import mavenn
import pdb
import pickle

# Imports from MAVE-NN
from mavenn.src.error_handling import handle_errors, check
from mavenn.src.validate import validate_1d_array, validate_nd_array

@handle_errors
def load(filename, verbose=True):

        """
        Method that will load a mave-nn model

        parameters
        ----------
        filename: (str)
            Filename of saved model.

        verbose: (bool)
            Whether to provide user feedback.

        returns
        -------
        loaded_model (mavenn-Model object)
            The model object that can be used to make predictions etc.

        """

        # Load model
        filename_pickle = filename + '.pickle'
        with open(filename_pickle, 'rb') as f:
            config_dict = pickle.load(f)

        # Create model object
        loaded_model = mavenn.Model(**config_dict['model_kwargs'])

        # Add in diffeomorphic mode fixing and standardization params
        loaded_model.unfixed_phi_mean = config_dict.get('unfixed_phi_mean', 0)
        loaded_model.unfixed_phi_std = config_dict.get('unfixed_phi_std', 1)
        loaded_model.y_mean = config_dict.get('y_mean', 0)
        loaded_model.y_std = config_dict.get('y_std', 1)
        loaded_model.x_stats = config_dict.get('x_stats', {})
        loaded_model.history = config_dict.get('history', None)
        loaded_model.info_for_layers_dict = \
            config_dict.get('info_for_layers_dict', {})

        # Load and set weights
        filename_h5 = filename + '.h5'
        loaded_model.get_nn().load_weights(filename_h5)

        # Provide feedback
        if verbose:
            print(f'Model loaded from these files:\n'
                  f'\t{filename_pickle}\n'
                  f'\t{filename_h5}')

        # Return model
        return loaded_model


@handle_errors
def vec_data_to_mat_data(y_n,
                         ct_n=None,
                         x_n=None):
    """

    Function to transform from vector to matrix format for MPA
    regression and MPA model evaluation.

    parameters
    ----------

    y_n: (np.ndarray)
        Array of N bin numbers y. Must be set by user.

    ct_n: (np.ndarray)
        Array N counts, one for each (sequence,bin) pair.
        If None, a value of 1 will be assumed for all observations

    x_n: (np.ndarray)
        List of N sequences. If None, each y_n will be
        assumed to come from a unique sequence.

    returns
    -------

    ct_my: (2D array of ints)
        Matrix of counts.

    x_m: (array)
        Corresponding list of x-values.
    """

    # Note: this use of validate_1d_array is needed to avoid a subtle
    # bug that occurs when inputs are pandas series with non-continguous
    # indices
    y_n = validate_1d_array(y_n).astype(int)
    N = len(y_n)
    if x_n is not None:
        x_n = validate_1d_array(x_n)
    else:
        x_n = np.arange(N)

    if ct_n is not None:
        ct_n = validate_1d_array(ct_n).astype(int)
    else:
        ct_n = np.ones(N).astype(int)

    # Cast y as array of ints
    y_n = np.array(y_n).astype(int)
    N = len(x_n)

    # This case is only for loading data. Should be tested/made more robust
    if N == 1:
        # should do check like check(mavenn.load_model==True,...

        return y_n.reshape(-1, y_n.shape[0]), x_n

    # Create dataframe
    data_df = pd.DataFrame()
    data_df['x'] = x_n
    data_df['y'] = y_n
    data_df['ct'] = ct_n

    # Sum over repeats
    data_df = data_df.groupby(['x', 'y']).sum().reset_index()

    # Pivot dataframe
    data_df = data_df.pivot(index='x', columns='y', values='ct')
    data_df = data_df.fillna(0).astype(int)

    # Clean dataframe
    data_df.reset_index(inplace=True)
    data_df.columns.name = None

    # Get ct_my values
    cols = [c for c in data_df.columns if not c in ['x']]

    ct_my = data_df[cols].values.astype(int)

    # Get x_m values
    x_m = data_df['x'].values

    return ct_my, x_m


@handle_errors
def mat_data_to_vec_data(ct_my,
                         x_m=None):
    """

    Function to transform from matrix format to vector format for MPA
    regression and MPA model evaluation.

    parameters
    ----------

    ct_my: (2D array of ints)
        Matrix of counts.

    x_m: (array)
        Corresponding list of x-values.

    parameters
    ----------

    y_n: (np.ndarray)
        Array of N bin numbers y. Must be set by user.

    ct_n: (np.ndarray)
        Array N counts, one for each (sequence,bin) pair.
        If None, a value of 1 will be assumed for all observations

    x_n: (np.ndarray)
        List of N sequences. If None, each y_n will be
        assumed to come from a unique sequence.

    """

    # Note: this use of validate_1d_array is needed to avoid a subtle
    # bug that occurs when inputs are pandas series with non-continguous
    # indices
    ct_my = validate_nd_array(ct_my).astype(int)
    check(ct_my.ndim == 2,
          f'ct_my.ndim={ct_my.ndim}; must be 2.')
    M, Y = ct_my.shape

    if x_m is not None:
        x_m = validate_1d_array(x_m)
    else:
        x_m = np.arange(M)

    # Create dataframe
    data_df = pd.DataFrame()
    y_cols = list(range(Y))
    data_df.loc[:, 'x'] = x_m
    data_df.loc[:, y_cols] = ct_my

    # Melt dataframe
    data_df = data_df.melt(id_vars='x',
                           value_vars=y_cols,
                           value_name='ct',
                           var_name='y')

    # Remove zero count rows
    ix = data_df['ct'] > 0
    data_df = data_df[ix].reset_index()
    data_df.sort_values(by=['ct','y'], ascending=False, inplace=True)

    # Get return values values
    x_n = data_df['x'].values
    y_n = data_df['y'].values
    ct_n = data_df['ct'].values

    return y_n, ct_n, x_n