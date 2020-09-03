from __future__ import print_function   # so that print behaves like python 3.x not a special lambda statement

import sys
sys.path.insert(0, '../../')
import mavenn

from mavenn.src.utils import get_example_dataset
from mavenn.src.validate import validate_alphabet
from mavenn.src.utils import get_1pt_variants

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

global_success_counter = 0
global_fail_counter = 0

# Common success and fail lists
bool_fail_list = [0, -1, 'True', 'x', 1]
bool_success_list = [False, True]


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
    #print('Test # %d: ' % test_num)

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


# Need to incorporate into model before testing.
def test_validate_alphabet():
    """ 20.09.10 JBK """

    # Tests that should pass
    success_list = [
        'dna',
        'rna',
        'protein',
        'protein*',
        np.array(['A', 'B', 'C']),
        {'A', 'B', 'C'},
        ['A', 'B', 'C'],
        pd.Series(['A', 'B', 'C'])
    ]

    # Tests that should fail
    fail_list = [
        'xna',
        'protein-',
        ['A','B','A'],
        [],
        {'A':5},
        np.array([['A','B'],['C','D']]),
        np.arange(5),
        pd.Series([])
    ]

    # Run tests of validate_alphabet
    test_parameter_values(func=validate_alphabet,
                          var_name='alphabet',
                          fail_list=fail_list,
                          success_list=success_list)

def test_get_1pt_variants():
    """20.09.01 JBK"""

    # Tests with alphabet='protein'
    test_parameter_values(func=get_1pt_variants,
                          var_name='wt_seq',
                          success_list=['QYKL'],
                          fail_list=['ACGU', 'QYKL*', '',
                                     0, ['A', 'C', 'G', 'T']],
                          alphabet='protein')

    # Tests with wt_seq='QYKL'
    test_parameter_values(func=get_1pt_variants,
                          var_name='alphabet',
                          success_list=['protein','protein*',
                                        ['Q', 'Y', 'K', 'L']],
                          fail_list=['dna','rna','ACGU','',0,
                                     ['Q', 'Y', 'K'], ['A', 'C', 'G', 'T']],
                          wt_seq='QYKL')

    # Test include_wt
    test_parameter_values(func=get_1pt_variants,
                          var_name='include_wt',
                          success_list=[True, False],
                          fail_list=[0, None],
                          wt_seq='QYKL', alphabet='protein')

    # Singleton tests
    test_parameter_values(func=get_1pt_variants,
                          var_name='wt_seq',
                          success_list=['ACGT'],
                          fail_list=['ACGU'],
                          alphabet='dna')
    test_parameter_values(func=get_1pt_variants,
                          var_name='wt_seq',
                          success_list=['ACGU'],
                          fail_list=['ACGT'],
                          alphabet='rna')
    test_parameter_values(func=get_1pt_variants,
                          var_name='wt_seq',
                          success_list=['QYKL*', 'QYKL'],
                          fail_list=['ACGU'],
                          alphabet='protein*')


def test_GlobalEpistasisModel():

    # load MPSA dataset for testing
    x, y = get_example_dataset()

    # test on subset of sequences
    x = x[0:1000].copy()
    y = y[0:1000].copy()

    # sequences arrays that fail when entered into mavenn.
    bad_x = 'x'

    # could possibly check if all elements are numeric
    # but that could slow things down
    bad_y = [1, 3, -2, 4.5]
    # also could check for nan's like np.isnan(bad_y).all()

    # test sequences parameter X
    test_parameter_values(func=mavenn.Model, var_name='x', fail_list=[bad_x], success_list=[x],
                          gpmap_type='additive', y=y, regression_type='GE', alphabet='rna')

    # test labels parameter y
    test_parameter_values(func=mavenn.Model, var_name='y', fail_list=[bad_y], success_list=[y],
                          gpmap_type='additive', x=x, regression_type='GE', alphabet='rna')

    # test labels parameter regression_type
    test_parameter_values(func=mavenn.Model, var_name='regression_type', fail_list=['polynomial'], success_list=['GE'],
                          gpmap_type='additive', x=x, y=y, alphabet='rna')

    # test labels parameter ge_nonlinearity_monotonic
    test_parameter_values(func=mavenn.Model, var_name='ge_nonlinearity_monotonic', fail_list=['True', -1],
                          success_list=[True,False], regression_type='GE',
                          gpmap_type='additive', x=x, y=y, alphabet='rna')

    # test labels parameter ge_nonlinearity_hidden_nodes
    test_parameter_values(func=mavenn.Model, var_name='ge_nonlinearity_hidden_nodes', fail_list=[0.6,-1,0],
                          success_list=[1,10,100],  regression_type='GE',
                          gpmap_type='additive', x=x, y=y, alphabet='rna')

    # test parameter gpmap_type
    test_parameter_values(func=mavenn.Model, var_name='gpmap_type', fail_list=['standard'],
                          success_list=['additive', 'neighbor', 'pairwise'],
                          regression_type='GE', x=x, y=y, alphabet='rna')

    # test parameter ge_heteroskedasticity_order
    test_parameter_values(func=mavenn.Model, var_name='ge_heteroskedasticity_order', fail_list=['0', 0.1, -1],
                          success_list=[0, 1, 10], gpmap_type='additive',
                          regression_type='GE', x=x, y=y, alphabet='rna')

    # test parameter theta_regularization
    test_parameter_values(func=mavenn.Model, var_name='theta_regularization', fail_list=['0', -1, -0.1],
                          success_list=[0, 0.1, 10], gpmap_type='additive',
                          regression_type='GE', x=x, y=y, alphabet='rna')

    # test parameter eta_regularization
    test_parameter_values(func=mavenn.Model, var_name='eta_regularization', fail_list=['0', -1, -0.1],
                          success_list=[0, 0.1, 10], gpmap_type='additive',
                          regression_type='GE', x=x, y=y, alphabet='rna')

    # test parameter ohe_batch_size
    test_parameter_values(func=mavenn.Model, var_name='ohe_batch_size', fail_list=['0', -1, -0.1, 0],
                          success_list=[20000], gpmap_type='additive',
                          regression_type='GE', x=x, y=y, alphabet='rna')


    '''
    # TODO: need to implement alphabet_dict checks in UI for GE and NA.
    # the following needs to be fixed in UI
    # test labels parameter alphabet
    test_parameter_values(func=mavenn.Model, var_name='alphabet', fail_list=['dna, protein'],
                          success_list=['rna'], model_type='additive',
                          regression_type='GE', x=x, y=y)
    '''

def test_NoiseAgnosticModel():

    # load MPSA dataset for testing
    x, y, ct_n = get_example_dataset(name='Sort-Seq')

    # test on subset of sequences
    x = x[0:1000].copy()
    y = y[0:1000].copy()
    ct_n = ct_n[0:1000].copy()

    # sequences arrays that fail when entered into mavenn.
    bad_X = 'x'

    # could possibly check if all elements are numeric
    # but that could slow things down
    bad_y = [[1, 3, -2, 4.5]]
    # Need to check for nans in y

    # test sequences parameter X
    test_parameter_values(func=mavenn.Model, var_name='x', fail_list=[bad_X], success_list=[x],
                          gpmap_type='additive', y=y, regression_type='MPA', alphabet='dna', ct_n=ct_n)

    # TODO: need to fix vec_data_to_mat_data to work with one example before using this test.
    # # test labels parameter y
    # test_parameter_values(func=mavenn.Model, var_name='y', fail_list=[bad_y], success_list=[y],
    #                       gpmap_type='additive', x=x, regression_type='MPA', alphabet='dna', ct_n=ct_n)

    # test labels parameter regression_type
    test_parameter_values(func=mavenn.Model, var_name='regression_type', fail_list=['polynomial'], success_list=['MPA'],
                          gpmap_type='additive', x=x, y=y, alphabet='dna', ct_n=ct_n)

    # test labels parameter gpmap_type
    test_parameter_values(func=mavenn.Model, var_name='gpmap_type', fail_list=['standard'],
                          success_list=['additive', 'neighbor', 'pairwise'],
                          regression_type='MPA', x=x, y=y, alphabet='dna', ct_n=ct_n)

    # test parameter na_hidden_nodes
    test_parameter_values(func=mavenn.Model, var_name='na_hidden_nodes', fail_list=['0', 0.1, -1, 0],
                          success_list=[1, 10], gpmap_type='additive', ct_n=ct_n,
                          regression_type='MPA', x=x, y=y, alphabet='dna')

    # test parameter theta_regularization
    test_parameter_values(func=mavenn.Model, var_name='theta_regularization', fail_list=['0', -1, -0.1],
                          success_list=[0, 0.1, 10], gpmap_type='additive', ct_n=ct_n,
                          regression_type='MPA', x=x, y=y, alphabet='dna')

    # test parameter ohe_batch_size
    test_parameter_values(func=mavenn.Model, var_name='ohe_batch_size', fail_list=['0', -1, -0.1, 0],
                          success_list=[20000], gpmap_type='additive', ct_n=ct_n,
                          regression_type='MPA', x=x, y=y, alphabet='dna')

def test_p_of_y_given_yhat():

    # Load GE model
    model_file = mavenn.__path__[0] + '/examples/models/ge_gaussian_GB1_model'
    model = mavenn.load(model_file)

def test_p_of_y_given_yhat():

    # Load GE model
    model_file = mavenn.__path__[0] + '/examples/models/ge_gaussian_GB1_model'
    model = mavenn.load(model_file)




    print('Model loaded!', model)


def run_tests():
    """
    Run all mavenn functional tests.

    parameters
    ----------
    None.

    return
    ------
    None.
    """

    test_validate_alphabet()

    test_get_1pt_variants()

    test_GlobalEpistasisModel()

    test_NoiseAgnosticModel()

