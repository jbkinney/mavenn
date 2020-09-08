# Standard imports
import numpy as np
import pandas as pd
import glob

# MAVE-NN imports
import mavenn
from mavenn.src.examples import load_example_dataset
from mavenn.src.validate import validate_alphabet
from mavenn.src.landscape import get_1pt_variants
from mavenn.src.utils import load
from mavenn.src.sequence_features import additive_model_features, pairwise_model_features
from mavenn.src.error_handling import check, handle_errors
from mavenn.tests.testing_utils import test_parameter_values


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
    data_df = load_example_dataset('mpsa')
    #x, y = load_example_dataset()
    x = data_df['x'].values
    y = data_df['y'].values

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
    #x, y, ct_n = load_example_dataset(name='Sort-Seq')
    data_df = load_example_dataset('sortseq')
    x = data_df['x'].values
    y = data_df['y'].values
    ct_n = data_df['ct'].values

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


def test_load():

    """
    Method that tests the load method.
    """



    # this model is missing all the values
    bad_GE_model_1 = mavenn.__path__[0] +'/tests/model_files/test_GE_model_bad1'

    # this file is missing the value for sequence.
    bad_GE_model_2 = mavenn.__path__[0] +'/tests/model_files/test_GE_model_bad2'

    # Good GE model file
    good_GE_model = mavenn.__path__[0] +'/tests/model_files/test_GE_model_good'

    test_parameter_values(func=load,
                          var_name='filename',
                          fail_list=[bad_GE_model_1, bad_GE_model_2],
                          success_list=[good_GE_model])


@handle_errors
def test_x_to_phi(model, seq):
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
def test_x_to_yhat(model, seq):
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


def test_x_to_phi_or_yhat():
    mavenn_dir = mavenn.__path__[0]
    model_dir = f'{mavenn_dir}/examples/models/'

    mpa_model = mavenn.load(model_dir + 'sortseq_mpa_additive')
    mpa_seq = 'GGCTTTACACTTTATGCTTCCGGCTCGTATGTTGTGTGG'
    mpa_seq_gap = 'GGCTTTACAC-TTATGCTTCCGGCTCGTATGTTGTGTGG'
    ge_model = mavenn.load(model_dir + 'gb1_ge_additive')
    ge_seq = 'QYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'
    ge_seq_gap = 'QYKLILNGKTLK-ETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'

    test_parameter_values(test_x_to_phi,
                          var_name='seq',
                          success_list=[ge_seq],
                          fail_list=[mpa_seq, ge_seq_gap],
                          model=ge_model)

    test_parameter_values(test_x_to_phi,
                          var_name='seq',
                          success_list=[mpa_seq],
                          fail_list=[ge_seq, mpa_seq_gap],
                          model=mpa_model)

    test_parameter_values(test_x_to_yhat,
                          var_name='seq',
                          success_list=[ge_seq],
                          fail_list=[mpa_seq, ge_seq_gap],
                          model=ge_model)

    test_parameter_values(test_x_to_yhat,
                          var_name='seq',
                          success_list=[],
                          fail_list=[mpa_seq],
                          model=mpa_model)


@handle_errors
def _test_phi_calculation(model_file):
    # Load model (assumes .h5 extension)
    model = mavenn.load(model_file[:-3])

    # Get sequence
    seq = model.x[0]

    # Get alphabet
    alphabet = model.model.alphabet
    alphabet = validate_alphabet(alphabet)

    # Explain test to user
    print(
f"""\nTesting phi calcluation
model     : {model_file}
gpmap_type: {model.gpmap_type}
alphabet  : {model.alphabet}
seq       : {seq}""")

    # Get MPA model parameters
    tmp_df = model.get_gpmap_parameters()

    # Create theta_df
    theta_df = pd.DataFrame()
    theta_df['id'] = [name.split('_')[1] for name in tmp_df['name']]
    theta_df['theta'] = tmp_df['value']
    theta_df.set_index('id', inplace=True)
    theta_df.head()

    # Get model type
    if model.gpmap_type == 'additive':
        f = additive_model_features
    elif model.gpmap_type in ['pairwise', 'neighbor']:
        f = pairwise_model_features
    else:
        check(model.gpmap_type in ['additive', 'neighbor', 'pairwise'],
              'Unrecognized model.gpmap_type: {model.gpmap_type}')

    # Encode sequence features
    x, names = f([seq], alphabet=alphabet)

    # Create dataframe
    x_df = pd.DataFrame()
    x_df['id'] = [name.split('_')[1] for name in names]
    x_df['x'] = x[0, :]
    x_df.set_index('id', inplace=True)
    x_df.head()

    # Make sure theta_df and x_df have the same indices
    x_ids = set(x_df.index)
    theta_ids = set(theta_df.index)
    check(x_ids >= theta_ids, f"theta features are not contained within x features.")

    # Merge theta_df and x_df into one dataframe
    df = pd.merge(left=theta_df, right=x_df, left_index=True, right_index=True,
                  how='left')

    # Make sure there are no nan entries
    num_null_entries = df.isnull().sum().sum()
    check(num_null_entries == 0,
          f'x_df and theta_df do not agree; found {num_null_entries} null entries.')

    # Compute phi from manual calculation
    phi_check = np.sum(df['theta'] * df['x'])

    # Compute phi using model method
    phi_model = model.x_to_phi(seq)

    check(np.isclose(phi_check, phi_model),
          f'phi_check: {phi_check} != phi_model: {phi_model} for gpmap_type: {model.gpmap_type}')
    print(
f"""phi_model : {phi_model}
phi_check : {phi_check}""")


def test_phi_calculations():
    mavenn_dir = mavenn.__path__[0]
    model_dir = f'{mavenn_dir}/examples/models/'

    # Get list of models in directory
    model_files = glob.glob(model_dir + '*.h5')

    test_parameter_values(func=_test_phi_calculation,
                          var_name='model_file',
                          success_list=model_files,
                          fail_list=[])
