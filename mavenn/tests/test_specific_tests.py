# Standard imports
import numpy as np
import pandas as pd
import glob
import pdb
import inspect
import pytest
# MAVE-NN imports
import mavenn
from mavenn.src.examples import load_example_dataset, load_example_model
from mavenn.src.validate import validate_alphabet
from mavenn.src.utils import load
from mavenn.src.error_handling import check, handle_errors
from mavenn.tests.testing_utils import test_parameter_values, generate_id
from mavenn.tests.define_tests import (
    get_validate_alphabet_tests,
    get_GlobalEpistasisModel_tests,
    get_NoiseAgnosticModel_tests,
    get_load_tests,
    get_x_to_phi_or_yhat_tests,
    get_GE_fit_tests,
    get_MPA_fit_tests,
    get_heatmap_tests
)



# Need to incorporate into model before testing.
# parametrize validate_alphabet tests
@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_validate_alphabet_tests(), ids=generate_id)
def test_validate_alphabet(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)


# def test_get_1pt_variants():
#     """20.09.01 JBK"""
#
#     # Tests with alphabet='protein'
#     test_parameter_values(func=get_1pt_variants,
#                           var_name='wt_seq',
#                           success_list=['QYKL'],
#                           fail_list=['ACGU', 'QYKL*', '',
#                                      0, ['A', 'C', 'G', 'T']],
#                           alphabet='protein')
#
#     # Tests with wt_seq='QYKL'
#     test_parameter_values(func=get_1pt_variants,
#                           var_name='alphabet',
#                           success_list=['protein','protein*',
#                                         ['Q', 'Y', 'K', 'L']],
#                           fail_list=['dna','rna','ACGU','',0,
#                                      ['Q', 'Y', 'K'], ['A', 'C', 'G', 'T']],
#                           wt_seq='QYKL')
#
#     # Test include_wt
#     test_parameter_values(func=get_1pt_variants,
#                           var_name='include_wt',
#                           success_list=[True, False],
#                           fail_list=[0, None],
#                           wt_seq='QYKL', alphabet='protein')
#
#     # Singleton tests
#     test_parameter_values(func=get_1pt_variants,
#                           var_name='wt_seq',
#                           success_list=['ACGT'],
#                           fail_list=['ACGU'],
#                           alphabet='dna')
#     test_parameter_values(func=get_1pt_variants,
#                           var_name='wt_seq',
#                           success_list=['ACGU'],
#                           fail_list=['ACGT'],
#                           alphabet='rna')
#     test_parameter_values(func=get_1pt_variants,
#                           var_name='wt_seq',
#                           success_list=['QYKL*', 'QYKL'],
#                           fail_list=['ACGU'],
#                           alphabet='protein*')

@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_GlobalEpistasisModel_tests(), ids=generate_id)
def test_GlobalEpistasisModel(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_NoiseAgnosticModel_tests(), ids=generate_id)
def test_NoiseAgnosticModel(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_load_tests(), ids=generate_id)
def test_load(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_x_to_phi_or_yhat_tests(), ids=generate_id)
def test_x_to_phi_or_yhat(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_GE_fit_tests(), ids=generate_id)
def test_GE_fit(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_MPA_fit_tests(), ids=generate_id)
def test_MPA_fit(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_heatmap_tests(), ids=generate_id)
def test_heatmap(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)