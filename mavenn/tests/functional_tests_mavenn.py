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
    X, y = get_example_dataset()

    # sequences arrays that fail when entered into mavenn.
    bad_X = 'x'

    # could possibly check if all elements are numeric
    # but that could slow things down
    bad_y = [1, 3, -2, 4.5]
    # also could check for nan's like np.isnan(bad_y).all()

    # test sequences parameter X
    test_parameter_values(func=mavenn.Model, var_name='X', fail_list=[bad_X], success_list=[X],
                          model_type='additive', y=y, regression_type='GE', alphabet_dict='rna')

    # test labels parameter y
    test_parameter_values(func=mavenn.Model, var_name='y', fail_list=[bad_y], success_list=[y],
                          model_type='additive', X=X, regression_type='GE', alphabet_dict='rna')

    # test labels parameter regression_type
    test_parameter_values(func=mavenn.Model, var_name='regression_type', fail_list=['polynomial'], success_list=['GE'],
                          model_type='additive', X=X, y=y, alphabet_dict='rna')

    # test labels parameter model_type
    test_parameter_values(func=mavenn.Model, var_name='model_type', fail_list=['standard'],
                          success_list=['additive', 'neighbor', 'pairwise'],
                          regression_type='GE', X=X, y=y, alphabet_dict='rna')

    # TODO: need to implement alphabet_dict checks in UI for GE and NA.
    # the following needs to be fixed in UI
    # # test labels parameter alphabet_dict
    # test_parameter_values(func=mavenn.Model, var_name='alphabet_dict', fail_list=['dna, protein'],
    #                       success_list=['rna'], model_type='additive',
    #                       regression_type='GE', X=X, y=y)


def test_NoiseAgnosticModel():

    # load MPSA dataset for testing
    X, y = get_example_dataset(name='Sort-Seq')

    # sequences arrays that fail when entered into mavenn.
    bad_X = 'x'

    # could possibly check if all elements are numeric
    # but that could slow things down
    bad_y = [[1, 3, -2, 4.5]]
    # also could check for nan's like np.isnan(bad_y).all()

    # test sequences parameter X
    test_parameter_values(func=mavenn.Model, var_name='X', fail_list=[bad_X], success_list=[X],
                          model_type='additive', y=y, regression_type='NA', alphabet_dict='dna')

    # test labels parameter y
    test_parameter_values(func=mavenn.Model, var_name='y', fail_list=[bad_y], success_list=[y],
                          model_type='additive', X=X, regression_type='NA', alphabet_dict='dna')

    # test labels parameter regression_type
    test_parameter_values(func=mavenn.Model, var_name='regression_type', fail_list=['polynomial'], success_list=['NA'],
                          model_type='additive', X=X, y=y, alphabet_dict='dna')

    # test labels parameter model_type
    test_parameter_values(func=mavenn.Model, var_name='model_type', fail_list=['standard'],
                          success_list=['additive', 'neighbor', 'pairwise'],
                          regression_type='NA', X=X, y=y, alphabet_dict='dna')



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

    test_GlobalEpistasisModel()
    test_NoiseAgnosticModel()

    # Tests mavenn.src.validate.validate_alphabet()
    test_validate_alphabet()
