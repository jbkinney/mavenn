
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import pdb
import inspect
# MAVE-NN imports
import mavenn
from mavenn.src.examples import load_example_dataset, load_example_model
from mavenn.src.validate import validate_alphabet
from mavenn.src.utils import load
from mavenn.src.error_handling import check, handle_errors

def get_validate_alphabet_tests():
    """
    Return a list of tests for the validate_alphabet function, to be used test_specific_tests.py.

    :return: list of tuples, each containing a test parameter, a value to test, a boolean indicating whether the test should fail, and a dictionary of keyword arguments:
        (param, val, should_fail, kwargs)
    """ 
    return [
        (validate_alphabet, 'alphabet', 'xna', True, {}),
        (validate_alphabet, 'alphabet', 'protein-', True, {}),
        (validate_alphabet, 'alphabet', ['A','B','A'], True, {}),
        (validate_alphabet, 'alphabet', [], True, {}),
        (validate_alphabet, 'alphabet', {'A':5}, True, {}),
        (validate_alphabet, 'alphabet', np.array([['A','B'],['C','D']]), True, {}),
        (validate_alphabet, 'alphabet', np.arange(5), True, {}),    
        (validate_alphabet, 'alphabet', pd.Series([]), True, {}),
        (validate_alphabet, 'alphabet', 'dna', False, {}),
        (validate_alphabet, 'alphabet', 'rna', False, {}),
        (validate_alphabet, 'alphabet', 'protein', False, {}),
        (validate_alphabet, 'alphabet', 'protein*', False, {}),
        (validate_alphabet, 'alphabet', np.array(['A', 'B', 'C']), False, {}),
        (validate_alphabet, 'alphabet', {'A', 'B', 'C'}, False, {}),
        (validate_alphabet, 'alphabet', ['A', 'B', 'C'], False, {}),
        (validate_alphabet, 'alphabet', pd.Series(['A', 'B', 'C']), False, {}),
    ]



def get_GlobalEpistasisModel_tests():

    """
    Return a list of tests for GlobalEpistasisModel, to be used test_specific_tests.py.

    :return: list of tuples, each containing a test parameter, a value to test, a boolean indicating whether the test should fail, and a dictionary of keyword arguments:
        (param, val, should_fail, kwargs)
    """
    # load MPSA dataset for testing
    data_df = load_example_dataset('mpsa').iloc[0:200].copy()

    #x,y = load-example dataset()
    x = data_df['x'].values
    y = data_df['y'].values

    L=len(x[0])


    # sequences arrays that fail when entered into mavenn.
    bad_x = 'x'
    bad_y = [1, 3, -2, 4.5]

    # Prep model to test mavenn.Model child methods
    model = mavenn.Model(regression_type='GE', L=L, alphabet='rna')
    model.set_data(x=x, y=y, verbose=False)
    model.fit(epochs=1, verbose=False)
    return [
        # test labels pareameter regression_type
        (mavenn.Model, 'regression_type', 'polynomial', True, {'gpmap_type': 'additive', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'regression_type', 'GE', False, {'gpmap_type': 'additive', 'alphabet': 'rna', 'L':L}),

        # test labels parameter ge_nonlinearity_monotonic
        (mavenn.Model, 'ge_nonlinearity_monotonic', 'True', True, {'regression_type': 'GE', 'gpmap_type': 'additive', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'ge_nonlinearity_monotonic', -1, True, {'regression_type': 'GE', 'gpmap_type': 'additive', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'ge_nonlinearity_monotonic', True, False, {'regression_type': 'GE', 'gpmap_type': 'additive', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'ge_nonlinearity_monotonic', False, False, {'regression_type': 'GE', 'gpmap_type': 'additive', 'alphabet': 'rna', 'L':L}),

        # test labels parameter ge_nonlinearity_hidden_nodes
        (mavenn.Model, 'ge_nonlinearity_hidden_nodes', 0.6, True, {'regression_type': 'GE', 'gpmap_type': 'additive', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'ge_nonlinearity_hidden_nodes', -1, True, {'regression_type': 'GE', 'gpmap_type': 'additive', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'ge_nonlinearity_hidden_nodes', 0, True, {'regression_type': 'GE', 'gpmap_type': 'additive', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'ge_nonlinearity_hidden_nodes', 1, False, {'regression_type': 'GE', 'gpmap_type': 'additive', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'ge_nonlinearity_hidden_nodes', 10, False, {'regression_type': 'GE', 'gpmap_type': 'additive', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'ge_nonlinearity_hidden_nodes', 100, False, {'regression_type': 'GE', 'gpmap_type': 'additive', 'alphabet': 'rna', 'L':L}),

        # test parameter gpmap_type
        (mavenn.Model, 'gpmap_type', 'standard', True, {'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'gpmap_type', 'additive', False, {'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'gpmap_type', 'neighbor', False, {'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'gpmap_type', 'pairwise', False, {'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),

        # test parameter ge_heteroskedascity_order
        (mavenn.Model, 'ge_heteroskedasticity_order', '0', True, {'gpmap_type': 'additive', 'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'ge_heteroskedasticity_order', 0.1, True, {'gpmap_type': 'additive', 'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'ge_heteroskedasticity_order', -1, True, {'gpmap_type': 'additive', 'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'ge_heteroskedasticity_order', 0, False, {'gpmap_type': 'additive', 'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'ge_heteroskedasticity_order', 1, False, {'gpmap_type': 'additive', 'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'ge_heteroskedasticity_order', 10, False, {'gpmap_type': 'additive', 'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),

        # test parameter eta_regularization
        (mavenn.Model, 'eta_regularization', '0', True, {'gpmap_type': 'additive', 'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'eta_regularization', -1, True, {'gpmap_type': 'additive', 'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'eta_regularization', -0.1, True, {'gpmap_type': 'additive', 'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'eta_regularization', 0, False, {'gpmap_type': 'additive', 'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'eta_regularization', 0.1, False, {'gpmap_type': 'additive', 'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'eta_regularization', 10, False, {'gpmap_type': 'additive', 'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),

        # test parameter ohe_batch_size
        (mavenn.Model, 'ohe_batch_size', '0', True, {'gpmap_type': 'additive', 'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'ohe_batch_size', -1, True, {'gpmap_type': 'additive', 'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'ohe_batch_size', -0.1, True, {'gpmap_type': 'additive', 'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'ohe_batch_size', 0, True, {'gpmap_type': 'additive', 'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),
        (mavenn.Model, 'ohe_batch_size', 20000, False, {'gpmap_type': 'additive', 'regression_type': 'GE', 'alphabet': 'rna', 'L':L}),
        
                # test model.simulate_method parameter N
        #test_parameter_values(func=model.simulate_dataset, var_name='N', fail_list=['0', -1, -0.1, 0],
        #                      success_list=[10, 1000])
                                
        # test model.simulate_method parameter x
        #test_parameter_values(func=model.simulate_dataset, var_name='x', fail_list=['0', -1, -0.1, 0],
        #                      success_list=[x[0:5]])                          

        # test model.simulate_method parameter validation_frac
        #test_parameter_values(func=model.simulate_dataset, var_name='validation_frac', fail_list=['0', -1, -0.1, 0],
        #                      success_list=[0.5], N=10)

        # test model.simulate_method parameter test_frac
        #test_parameter_values(func=model.simulate_dataset, var_name='test_frac', fail_list=['0', -1, -0.1, 0],
        #                      success_list=[0.5], N=10)

        # TODO: using gauge='user' breaks, need to test with p_lc, and x_wt
        # test model.get_theta
        (model.get_theta, 'gauge', 0, True, {}),
        (model.get_theta, 'gauge', 'lorentz', True, {}),
        (model.get_theta, 'gauge', "none", False, {}),
        (model.get_theta, 'gauge', "uniform", False, {}),
        (model.get_theta, 'gauge', "empirical", False, {}),
        (model.get_theta, 'gauge', "consensus", False, {}),

    ]



def get_NoiseAgnosticModel_tests():
    """
    Return a list of tests for NoiseAgnosticModel, to be used test_specific_tests.py.

    :return: list of tuples, each containing a test parameter, a value to test, a boolean indicating whether the test should fail, and a dictionary of keyword arguments:
        (param, val, should_fail, kwargs)
    """
    data_df = load_example_dataset('sortseq').iloc[0:200].copy()
    x = data_df['x'].values
    y = data_df.filter(regex='ct_*').values
    L = len(x[0])
    Y = 10
    bad_x = 'x'
    
    # could possibly check if all elements are numeric
    # but that could slow things down
    bad_y = [[1, 3, -2, 4.5]]
    # Need to check for nans in y
    return [
            # test sequences parameter X
        # test_parameter_values(func=mavenn.Model, var_name='x', fail_list=[bad_X], success_list=[x],
        #                       gpmap_type='additive', y=y, regression_type='MPA', alphabet='dna', ct_n=ct_n)

            # test labels parameter regression_type
        (mavenn.Model, 'regression_type', 'polynomial', True, {'gpmap_type': 'additive', 'alphabet': 'dna', 'L':L, 'Y':Y}),
        (mavenn.Model, 'regression_type', 'MPA', False, {'gpmap_type': 'additive', 'alphabet': 'dna', 'L':L, 'Y':Y}),

        # test labels parameter gpmap_type
        (mavenn.Model, 'gpmap_type', 'standard', True, {'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),
        (mavenn.Model, 'gpmap_type', 'additive', False, {'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),
        (mavenn.Model, 'gpmap_type', 'neighbor', False, {'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),
        (mavenn.Model, 'gpmap_type', 'pairwise', False, {'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),

        # test parameter mpa_hidden_nodes
        (mavenn.Model, 'mpa_hidden_nodes', 0, True, {'gpmap_type': 'additive', 'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),
        (mavenn.Model, 'mpa_hidden_nodes', 0.1, True, {'gpmap_type': 'additive', 'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),
        (mavenn.Model, 'mpa_hidden_nodes', -1, True, {'gpmap_type': 'additive', 'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),
        (mavenn.Model, 'mpa_hidden_nodes', 0, True, {'gpmap_type': 'additive', 'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),
        (mavenn.Model, 'mpa_hidden_nodes', 1, False, {'gpmap_type': 'additive', 'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),
        (mavenn.Model, 'mpa_hidden_nodes', 10, False, {'gpmap_type': 'additive', 'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),
        
        # test parameter theta_regularization
        (mavenn.Model, 'theta_regularization', '0', True, {'gpmap_type': 'additive', 'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),
        (mavenn.Model, 'theta_regularization', -1, True, {'gpmap_type': 'additive', 'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),
        (mavenn.Model, 'theta_regularization', -0.1, True, {'gpmap_type': 'additive', 'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),
        (mavenn.Model, 'theta_regularization', 0, False, {'gpmap_type': 'additive', 'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),
        (mavenn.Model, 'theta_regularization', 0.1, False, {'gpmap_type': 'additive', 'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),
        (mavenn.Model, 'theta_regularization', 10, False, {'gpmap_type': 'additive', 'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),

        # test parameter ohe_batch_size
        (mavenn.Model, 'ohe_batch_size', '0', True, {'gpmap_type': 'additive', 'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),
        (mavenn.Model, 'ohe_batch_size', -1, True, {'gpmap_type': 'additive', 'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),
        (mavenn.Model, 'ohe_batch_size', -0.1, True, {'gpmap_type': 'additive', 'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),
        (mavenn.Model, 'ohe_batch_size', 0, True, {'gpmap_type': 'additive', 'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),
        (mavenn.Model, 'ohe_batch_size', 20000, False, {'gpmap_type': 'additive', 'regression_type': 'MPA', 'alphabet': 'dna', 'L':L, 'Y':Y}),

    ]


def get_load_tests():
    """
    Return a list of tests for the load method, to be used test_specific_tests.py.
    """
    maven_dir = mavenn.__path__[0]
    model_dir = os.path.join(maven_dir, 'examples', 'models')

    good_MPA_model_1 = os.path.join(model_dir, 'sortseq_mpa_additive')
    good_GE_model_1 = os.path.join(model_dir, 'gb1_ge_additive')

 # Good GE model file
    #good_GE_model_1 = mavenn.__path__[0] +'/tests/model_files/test_GE_model_good'
    #good_MPA_model_1 = mavenn.__path__[0] + '/tests/model_files/test_MPA_model_good'
    
    return [
        (mavenn.load, 'filename', good_GE_model_1, False, {}),
        (mavenn.load, 'filename', good_MPA_model_1, False, {}),
        
    ]

@handle_errors
def run_test_x_to_phi(model,seq):
    x = seq
    phi = model.x_to_phi(x)
    check(isinstance(phi, float), f'phi is {type(phi)}, not a float')
    check(np.isfinite(phi), f'phi={phi} is not finite.')

    x = np.array(seq)
    phi = model.x_to_phi(x)
    check(isinstance(phi, float), f'phi is {type(phi)}, not a float')

    x = [seq,
         seq,
         seq]
    shape = (3,)
    phi = model.x_to_phi(x)
    check(isinstance(phi, np.ndarray), f'phi is {type(phi)}, not a np.ndarray')
    check(phi.shape == shape,
          f'phi={phi} does not have the expected shape={shape}')

    x = [[seq, seq, seq]]
    shape = (1, 3)
    phi = model.x_to_phi(x)
    check(isinstance(phi, np.ndarray), f'phi is {type(phi)}, not a np.ndarray')
    check(phi.shape == shape,
          f'phi={phi} does not have the expected shape={shape}')

    x = [[seq],
         [seq],
         [seq]]
    shape = (3, 1)
    phi = model.x_to_phi(x)
    check(isinstance(phi, np.ndarray), f'phi is {type(phi)}, not a np.ndarray')
    check(phi.shape == shape,
          f'phi={phi} does not have the expected shape={shape}')

    x = [[[seq],
          [seq],
          [seq]]]
    shape = (1, 3, 1)
    phi = model.x_to_phi(x)
    check(isinstance(phi, np.ndarray), f'phi is {type(phi)}, not a np.ndarray')
    check(phi.shape == shape,
          f'phi={phi} does not have the expected shape={shape}')

    x = np.array([seq, seq, seq])
    shape = (3,)
    phi = model.x_to_phi(x)
    check(isinstance(phi, np.ndarray), f'phi is {type(phi)}, not a np.ndarray')
    check(phi.shape == shape,
          f'phi={phi} does not have the expected shape={shape}')
    
@handle_errors
def run_test_x_to_yhat(model, seq):
    x = seq
    yhat = model.x_to_yhat(x)
    check(isinstance(yhat, float), f'yhat is {type(yhat)}, not a float')
    check(np.isfinite(yhat), f'yhat={yhat} is not finite.')

    x = np.array(seq)
    yhat = model.x_to_yhat(x)
    check(isinstance(yhat, float), f'yhat is {type(yhat)}, not a float')

    x = [seq,
         seq,
         seq]
    shape = (3,)
    yhat = model.x_to_yhat(x)
    check(isinstance(yhat, np.ndarray),
          f'yhat is {type(yhat)}, not a np.ndarray')
    check(yhat.shape == shape,
          f'yhat={yhat} does not have the expected shape={shape}')

    x = [[seq, seq, seq]]
    shape = (1, 3)
    yhat = model.x_to_yhat(x)
    check(isinstance(yhat, np.ndarray),
          f'yhat is {type(yhat)}, not a np.ndarray')
    check(yhat.shape == shape,
          f'yhat={yhat} does not have the expected shape={shape}')

    x = [[seq],
         [seq],
         [seq]]
    shape = (3, 1)
    yhat = model.x_to_yhat(x)
    check(isinstance(yhat, np.ndarray),
          f'yhat is {type(yhat)}, not a np.ndarray')
    check(yhat.shape == shape,
          f'yhat={yhat} does not have the expected shape={shape}')

    x = [[[seq],
          [seq],
          [seq]]]
    shape = (1, 3, 1)
    yhat = model.x_to_yhat(x)
    check(isinstance(yhat, np.ndarray),
          f'yhat is {type(yhat)}, not a np.ndarray')
    check(yhat.shape == shape,
          f'yhat={yhat} does not have the expected shape={shape}')

    x = np.array([seq, seq, seq])
    shape = (3,)
    yhat = model.x_to_yhat(x)
    check(isinstance(yhat, np.ndarray),
          f'yhat is {type(yhat)}, not a np.ndarray')
    check(yhat.shape == shape,
          f'yhat={yhat} does not have the expected shape={shape}')

def get_x_to_phi_or_yhat_tests():
    """
    Method that returns a list of tests for the x_to_phi and x_to_yhat methods of the Model class

    :return: list of tuples, each containing a test parameter, a value to test, a boolean indicating whether the test should fail, and a dictionary of keyword arguments:
        (param, val, should_fail, kwargs)
    """
    mavenn_dir = mavenn.__path__[0]
    model_dir = os.path.join(mavenn_dir, 'examples', 'models')

    mpa_model = load(os.path.join(model_dir, 'sortseq_mpa_additive'))
    mpa_seq = 'AATTAATGTGAGTTAGCTCACTCATTAGGCACCCCAGGCTTTACACTTTATGCTTCCGGCTCGTATGTTGTGTGG'
    mpa_seq_gap = 'AATTAATGTGAGTTAGCTCACTC--TAGGCACCCCAGGCTTTACACTTTATGCTTCCGGCTCGTATGTTGTGTGG'

    ge_model = load(os.path.join(model_dir, 'gb1_ge_additive'))
    ge_seq = 'QYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'
    ge_seq_gap = 'QYKLILNGKTLK-ETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'

    return [
        (run_test_x_to_phi, 'seq', mpa_seq, True, {'model': ge_model}),
        (run_test_x_to_phi, 'seq', ge_seq_gap, True, {'model': ge_model}),
        (run_test_x_to_phi, 'seq', ge_seq, False, {'model': ge_model}),

        (run_test_x_to_phi, 'seq', ge_seq, True, {'model': mpa_model}),
        (run_test_x_to_phi, 'seq', mpa_seq_gap, True, {'model': mpa_model}),
        (run_test_x_to_phi, 'seq', mpa_seq, False, {'model': mpa_model}),

        (run_test_x_to_yhat, 'seq', mpa_seq, True, {'model': ge_model}),
        (run_test_x_to_yhat, 'seq', ge_seq_gap, True, {'model': ge_model}),
        (run_test_x_to_yhat, 'seq', ge_seq, False, {'model': ge_model}),

        (run_test_x_to_yhat, 'seq', mpa_seq, True, {'model': mpa_model}),


    ]

@handle_errors
def run_test_for_nan_in_model_methods(model, seqs, y, regression_type):

    """
    Method that evaluates model methods and checks
    if are any NANs in the output.

    parameters
    ----------
    model: (mavenn model)
        Mavenn model object whose methods will be used
        to compute various outputs.

    seqs: (array-like of strings)
        Sequences which will be input to the methods x_to_*.

    y: (array-like of floats)
        Observations/y-values corresponding to the seqs parameter.

    regression_type: (str)
        String specifying 'GE' or 'MPA' regression.

    returns
    -------
    None.
    """

    # sum the arrays produced by the following Model
    # methods and then use np.isnan together with check.
    check(np.isnan(np.sum(model.x_to_phi(seqs))) == False,
          'x_to_phi produced a NAN')

    check(np.isnan(np.sum(model.p_of_y_given_phi(y=y,
                                                 phi=model.x_to_phi(seqs)).ravel())) == False,
          'p_of_y_given_phi produce a NAN')

    #I, dI = model.I_predictive(x=seqs, y=y)
    #check(np.isnan(I) == False, 'Predictive information computed to NAN')
    #check(np.isnan(dI) == False, 'Error predictive information computed to NAN')

    if regression_type == 'MPA':

        # TODO: this method's broadcasting doesn't work as in GE.
        # This should be updated to work with the new MPA format
        # For now, pick a specific bin_number to test for NANs.
        bin_number = 5
        check(np.isnan(np.sum(model.p_of_y_given_x(y=bin_number, x=seqs).ravel())) == False,
              'p_of_y_given_x produce a NAN')

    # method applicable only for GE regression.
    elif regression_type == 'GE':
        check(np.isnan(np.sum(model.x_to_yhat(seqs))) == False,
              'x_to_yhat produced a NAN')

        check(np.isnan(np.sum(model.yhat_to_yq(model.x_to_yhat(seqs)).ravel())) == False,
              'yhat_to_yq produce a NAN')

        check(np.isnan(np.sum(model.phi_to_yhat(model.x_to_phi(seqs)).ravel())) == False,
              'phi to yhat produced a NAN')

        check(np.isnan(np.sum(model.p_of_y_given_yhat(y=y, yhat=model.x_to_yhat(seqs)).ravel())) == False,
              'p_of_y_given_yhat produced a NAN')

        check(np.isnan(np.sum(model.p_of_y_given_x(y=y, x=seqs).ravel())) == False,
              'p_of_y_given_x produce a NAN')

@handle_errors
def get_GE_fit_tests():

    """
    Method that returns a list of tests for the fit method of the Model class
    for GE regression. Small subsets of data are used for
    training, for all combinations of gpmap_type(s) and
    GE noise models. Models are trained for one epoch
    and all Model method outputs are subsequently checked
    for NANs.

    parameters
    ----------
    None

    returns
    -------
    List of tuples, each containing a test parameter, a value to test, a boolean indicating whether the test should fail, and a dictionary of keyword arguments:
        (param, val, should_fail, kwargs)

    """

    # Turn off warnings/retracing just for testing,
    # however need to figure out how to properly
    # handle these warnings.
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    tests = []

    GE_datasets = ['mpsa', 'gb1']

    GE_noise_model_types = ['Gaussian', 'Cauchy', 'SkewedT']
    gpmap_types = ['additive', 'neighbor', 'pairwise']

    # loop over GE datasets for testing fit.
    for ge_dataset in GE_datasets:

        data_df = mavenn.load_example_dataset(ge_dataset)

        # use small subset of data for quick training
        data_df = data_df.loc[0:200].copy()

        # get training set from data_df
        ix = (data_df['set']!='test')
        L = len(data_df['x'][0])
        train_df = data_df[ix]
        test_df = data_df[~ix]

        # set alpbabet according to dataset.
        if ge_dataset == 'mpsa':
            alphabet = 'rna'
        elif ge_dataset == 'gb1':
            alphabet = 'protein'

        # loop over different gpmap_types
        for gpmap_type in gpmap_types:

            # loop over different GE noise model types
            for GE_noise_model_type in GE_noise_model_types:
                  #print(f'======= {gpmap_type} : {GE_noise_model_type} : {dataset} ========')

                  # Define model
                  model = mavenn.Model(regression_type='GE',
                                     L=L,
                                     alphabet=alphabet,
                                     gpmap_type=gpmap_type,
                                     ge_noise_model_type=GE_noise_model_type,
                                     ge_heteroskedasticity_order=2)

                  # Set training data
                  model.set_data(x=train_df['x'],
                                    y=train_df['y'],
                                    shuffle=True,
                                    verbose=False)

                  # Fit model to data
                  try:
                        _history = model.fit(epochs=1,
                                       linear_initialization=True,
                                       batch_size=200,
                                       verbose=False)
                  except AssertionError:
                        print('Debugging...')
                        raise AssertionError
      
                  # check model methods for NANs
                  #print('Check for NANs in the output of model methods')
                  print(f'Testing model inference with: \n'
                        f'gpmap_type={repr(gpmap_type)}, \n'
                        f'dataset={repr(ge_dataset)}, \n'
                        f'noise_model={repr(GE_noise_model_type)}')
                  
                  tests.append((run_test_for_nan_in_model_methods,'seqs', [np.nan], True, {'model': model, 'y': test_df['y'], 'regression_type': 'GE'}))
                  tests.append((run_test_for_nan_in_model_methods,'seqs', test_df['x'].values, False, {'model': model, 'y': test_df['y'], 'regression_type': 'GE'}))

    return tests


@handle_errors
def get_MPA_fit_tests():

    """
    Method that returns a list of tests for the fit method of the Model class
    for MPA regression. Small subsets of data are used for
    training, for all combinations of gpmap_type(s).
    Models are trained for one epoch and all Model
    method outputs are subsequently checked for NANs.

    parameters
    ----------
    None

    returns
    -------
    List of tuples, each containing a test parameter, a value to test, a boolean indicating whether the test should fail, and a dictionary of keyword arguments:
        (param, val, should_fail, kwargs)

    """

    # turn off warnings/retracing just for testing
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    tests = []

    gpmap_types = ['additive', 'neighbor', 'pairwise']

    data_df = mavenn.load_example_dataset('sortseq')

    # use small subset of data for quick training
    data_df = data_df.loc[0:200].copy()

    # Comptue sequence length and number of bins
    L = len(data_df['x'][0])
    y_cols = [c for c in data_df.columns if 'ct_' in c]
    Y = len(y_cols)
    print(f'L={L}, Y={Y}')

    # Split into trianing and test data
    ix = (data_df['set'] != 'test')
    L = len(data_df['x'][0])
    train_df = data_df[ix]
    test_df = data_df[~ix]

    # loop over different gpmap_types
    for gpmap_type in gpmap_types:

        # Define model
        model = mavenn.Model(regression_type='MPA',
                             L=L,
                             Y=Y,
                             alphabet='dna',
                             gpmap_type=gpmap_type)

        model.set_data(x=train_df['x'].values,
                       y=train_df[y_cols].values,
                       verbose=False)

        # Fit model to data
        history = model.fit(epochs=1,
                            batch_size=250,
                            verbose=False)
        # check model methods for NANs
        #print('Check for NANs in the output of model methods')
        print(f'gpmap_type = {gpmap_type}, dataset = sortseq')
        print(f'Testing model inference with: \n'
              f'gpmap_type={repr(gpmap_type)}, \n'
              f'dataset="sortseq"')
        
        tests.append((run_test_for_nan_in_model_methods,'seqs', [np.nan], True, {'model': model, 'y': test_df[y_cols].values, 'regression_type': 'MPA'}))
        tests.append((run_test_for_nan_in_model_methods,'seqs', test_df['x'].values, False, {'model': model, 'y': test_df[y_cols].values, 'regression_type': 'MPA'}))

    return tests

# @handle_errors
# def _test_phi_calculation(model_file):
#     # Load model (assumes .h5 extension)
#     model = mavenn.load(model_file[:-3])
#
#     # Get sequence
#     seq = model.x_stats['consensus_seq']
#
#     # Get alphabet
#     alphabet = model.model.alphabet
#     alphabet = validate_alphabet(alphabet)
#
#     # Explain test to user
#     print(
# f"""\nTesting phi calcluation
# model     : {model_file}
# gpmap_type: {model.gpmap_type}
# alphabet  : {model.alphabet}
# seq       : {seq}""")
#
#     # Get MPA model parameters
#     tmp_df = model.get_gpmap_parameters()
#
#     # Create theta_df
#     theta_df = pd.DataFrame()
#     theta_df['id'] = [name.split('_')[1] for name in tmp_df['name']]
#     theta_df['theta'] = tmp_df['value']
#     theta_df.set_index('id', inplace=True)
#     theta_df.head()
#
#     # Get model type
#     if model.gpmap_type == 'additive':
#         f = additive_model_features
#     elif model.gpmap_type in ['pairwise', 'neighbor']:
#         f = pairwise_model_features
#     else:
#         check(model.gpmap_type in ['additive', 'neighbor', 'pairwise'],
#               'Unrecognized model.gpmap_type: {model.gpmap_type}')
#
#     # Encode sequence features
#     x, names = f([seq], alphabet=alphabet)
#
#     # Create dataframe
#     x_df = pd.DataFrame()
#     x_df['id'] = [name.split('_')[1] for name in names]
#     x_df['x'] = x[0, :]
#     x_df.set_index('id', inplace=True)
#     x_df.head()
#
#     # Make sure theta_df and x_df have the same indices
#     x_ids = set(x_df.index)
#     theta_ids = set(theta_df.index)
#     check(x_ids >= theta_ids, f"theta features are not contained within x features.")
#
#     # Merge theta_df and x_df into one dataframe
#     df = pd.merge(left=theta_df, right=x_df, left_index=True, right_index=True,
#                   how='left')
#
#     # Make sure there are no nan entries
#     num_null_entries = df.isnull().sum().sum()
#     check(num_null_entries == 0,
#           f'x_df and theta_df do not agree; found {num_null_entries} null entries.')
#
#     # Compute phi from manual calculation
#     phi_check = np.sum(df['theta'] * df['x'])
#
#     # Compute phi using model method
#     phi_model = model.x_to_phi(seq)
#
#     check(np.isclose(phi_check, phi_model, atol=1E-5),
#           f'phi_check: {phi_check} != phi_model: {phi_model} for gpmap_type: {model.gpmap_type}')
#     print(
# f"""phi_model : {phi_model}
# phi_check : {phi_check}""")


# def test_phi_calculations():
#     mavenn_dir = mavenn.__path__[0]
#     model_dir = f'{mavenn_dir}/examples/models/'
#
#     # Get list of models in directory
#     model_files = glob.glob(model_dir + '*.h5')
#
#     test_parameter_values(func=_test_phi_calculation,
#                           var_name='model_file',
#                           success_list=model_files,
#                           fail_list=[])


# def test_load_example():
#
#     successful_which_list = [None, 'model', 'training_data', 'test_data']
#     fail_which_list = [0, 'xxx', True]
#
#     successful_dataset_names_list = [None, 'mpsa', 'sortseq', 'gb1']
#     incorrect_dataset_names_list = [0, 'xxx']
#
#     successful_model_names_list = ["gb1_ge_additive",
#                                    "mpsa_ge_pairwise",
#                                    "sortseq_mpa_additive"]
#
#     incorrect_model_names_list = [0, "gb1", 'xxx']
#
#     # test parameter which
#     test_parameter_values(func=load_example,
#                           var_name='which',
#                           success_list=successful_which_list,
#                           fail_list=fail_which_list)
#
#     # test parameter name, with which='test_data'
#     test_parameter_values(func=load_example,
#                           var_name='name',
#                           which='test_data',
#                           success_list=successful_dataset_names_list,
#                           fail_list=incorrect_dataset_names_list)
#
#     # test parameter name, with which='model'
#     test_parameter_values(func=load_example,
#                           var_name='name',
#                           which='model',
#                           success_list=successful_model_names_list,
#                           fail_list=incorrect_model_names_list)
#

def get_heatmap_tests():
    """
    Method that returns a list of tests for the heatmap method


    :return: list of tuples, each containing a test parameter, a value to test, a boolean indicating whether the test should fail, and a dictionary of keyword arguments:
        (param, val, should_fail, kwargs)
    """
    df = pd.DataFrame(columns=['x', 'y', 'z', 'q'], data=np.random.rand(10, 4))
    values = df.values
    alphabet = df.columns
    
    return [

        # Test df
        # test 1
        (mavenn.heatmap, 'df', None, True, {'values': None, 'alphabet': None}),
        (mavenn.heatmap, 'df', 1, True, {'values': None, 'alphabet': None}),
        (mavenn.heatmap, 'df', 'hi', True, {'values': None, 'alphabet': None}),
        (mavenn.heatmap, 'df', df, False, {'values': None, 'alphabet': None}),
        # test 2
        (mavenn.heatmap, 'df', df, True, {'values': values, 'alphabet': alphabet}),
        (mavenn.heatmap, 'df', 1, True, {'values': values, 'alphabet': alphabet}),
        (mavenn.heatmap, 'df', 'hi', True, {'values': values, 'alphabet': alphabet}),
        (mavenn.heatmap, 'df', None, False, {'values': values, 'alphabet': alphabet}),
        # test 3 
        (mavenn.heatmap, 'df', None, True, {'values': values, 'alphabet': None}),
        (mavenn.heatmap, 'df', df, True, {'values': values, 'alphabet': None}),
        (mavenn.heatmap, 'df', 1, True, {'values': values, 'alphabet': None}),
        (mavenn.heatmap, 'df', 'hi', True, {'values': values, 'alphabet': None}),
        # test 4
        (mavenn.heatmap, 'df', None, True, {'values': None, 'alphabet': alphabet}),
        (mavenn.heatmap, 'df', df, True, {'values': None, 'alphabet': alphabet}),
        (mavenn.heatmap, 'df', 1, True, {'values': None, 'alphabet': alphabet}),
        (mavenn.heatmap, 'df', 'hi', True, {'values': None, 'alphabet': alphabet}),

        # Test values
        # test 5
        (mavenn.heatmap, 'values', None, True, {'df': None, 'alphabet': alphabet}),
        (mavenn.heatmap, 'values', 1, True, {'df': None, 'alphabet': alphabet}),
        (mavenn.heatmap, 'values', 'hi', True, {'df': None, 'alphabet': alphabet}),
        (mavenn.heatmap, 'values', values[:,:-1], True, {'df': None, 'alphabet': alphabet}),
        (mavenn.heatmap, 'values', values, False, {'df': None, 'alphabet': alphabet}),
        (mavenn.heatmap, 'values', df.values, False, {'df': None, 'alphabet': alphabet}),

        # Test alphabet
        # test 6
        (mavenn.heatmap, 'alphabet', None, True, {'values': values, 'df': None}),
        (mavenn.heatmap, 'alphabet', 1, True, {'values': values, 'df': None}),
        (mavenn.heatmap, 'alphabet', 'hi', True, {'values': values, 'df': None}),
        (mavenn.heatmap, 'alphabet', df.columns, False, {'values': values, 'df': None}),
        (mavenn.heatmap, 'alphabet', list(df.columns), False, {'values': values, 'df': None}),
        (mavenn.heatmap, 'alphabet', 'dna', False, {'values': values, 'df': None}),
          

    ]
