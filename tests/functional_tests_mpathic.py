from __future__ import print_function   # so that print behaves like python 3.x not a special lambda statement
import mpathic as mpa
import mpathic
import numpy as np
from mpathic import shutthefuckup

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

def test_mpathic_io():

    ###########################
    ## io.load_dataset tests ##
    ###########################

    # good and bad arguments for load dataset in text form
    bad_file_arg_load_dataset_text_1 = "../../mpathic/MPAthic_tests/input/dataset_bad_badseqs.txt"
    bad_file_arg_load_dataset_text_2 = "../../mpathic/MPAthic_tests/input/dataset_bad_badcounts.txt"
    bad_file_arg_load_dataset_text_3 = "../../mpathic/MPAthic_tests/input/dataset_bad_badseqtype.txt"
    bad_file_arg_load_dataset_text_4 = "../../mpathic/MPAthic_tests/input/dataset_bad_floatcounts.txt"
    bad_file_arg_load_dataset_text_5 = "../../mpathic/MPAthic_tests/input/dataset_bad_wrongcolname.txt"

    good_file_arg_load_dataset_text_1 = "../../mpathic/MPAthic_tests/input/dataset_crp.txt"
    good_file_arg_load_dataset_text_2 = "../../mpathic/MPAthic_tests/input/dataset_good_noctcol.txt"
    good_file_arg_load_dataset_text_3 = "../../mpathic/MPAthic_tests/input/dataset_good_nonconsecutivebins.txt"
    good_file_arg_load_dataset_text_4 = "../../mpathic/MPAthic_tests/input/dataset_good_pro.txt"
    good_file_arg_load_dataset_text_5 = "../../mpathic/MPAthic_tests/input/dataset_good_rna.txt"
    good_file_arg_load_dataset_text_6 = "../../mpathic/MPAthic_tests/input/dataset_good.txt"
    good_file_arg_load_dataset_text_7 = "../../mpathic/MPAthic_tests/input/dataset_pro.txt"
    good_file_arg_load_dataset_text_8 = "../../mpathic/MPAthic_tests/input/dataset.txt"

    # test parameter file args for load_dataset for file_type = text
    test_parameter_values \
            (
            func=mpathic.io.load_dataset, var_name='file_arg',
            fail_list=[
                bad_file_arg_load_dataset_text_1,
                bad_file_arg_load_dataset_text_2,
                bad_file_arg_load_dataset_text_3,
                bad_file_arg_load_dataset_text_4,
                bad_file_arg_load_dataset_text_5],
            success_list=[
                good_file_arg_load_dataset_text_1,
                good_file_arg_load_dataset_text_2,
                good_file_arg_load_dataset_text_3,
                good_file_arg_load_dataset_text_4,
                good_file_arg_load_dataset_text_5,
                good_file_arg_load_dataset_text_6,
                good_file_arg_load_dataset_text_7,
                good_file_arg_load_dataset_text_8]
        )

    bad_file_arg_load_dataset_fasta_1 = "../../mpathic/MPAthic_tests/input/genome_ecoli_100lines_bad_char.fa"
    good_file_arg_load_dataset_fasta_1 = "../../mpathic/MPAthic_tests/input/genome_ecoli_100lines.fa"
    good_file_arg_load_dataset_fasta_2 = "../../mpathic/MPAthic_tests/input/bin_hbsites.fa"

    # test parameter file args for load_dataset for file_type = fasta, seq_type = 'dna'
    test_parameter_values(func=mpathic.io.load_dataset, var_name='file_arg',
                          fail_list=[bad_file_arg_load_dataset_fasta_1],
                          success_list=[good_file_arg_load_dataset_fasta_1, good_file_arg_load_dataset_fasta_2],
                          file_type='fasta', seq_type='dna')

    # test parameter file args for load_dataset for file_type = fastq, seq_type = 'dna'

    bad_file_arg_load_dataset_fastq_2 = "../../mpathic/MPAthic_tests/input/seq_bad_hasNs.fastq"
    good_file_arg_load_dataset_fastq_1 = "../../mpathic/MPAthic_tests/input/seq_good.fastq"

    # this file causes trouble: succeeds but should fail?
    bad_file_arg_load_dataset_fastq_1 = "../../mpathic/MPAthic_tests/input/seq_bad_actuallyfasta.fastq"

    # test parameter file args for load_dataset for file_type = fasta, seq_type = 'dna'
    test_parameter_values(func=mpathic.io.load_dataset, var_name='file_arg',
                          fail_list=[bad_file_arg_load_dataset_fastq_2],
                          success_list=[good_file_arg_load_dataset_fastq_1],
                          file_type='fastq', seq_type='dna')

    ###############################
    ## io.load_dataset tests end ##
    ###############################

    #########################
    ## io.load_model tests ##
    #########################

    bad_file_arg_load_model_1 = "../../mpathic/MPAthic_tests/input/model_bad_mat_badcol.txt"
    bad_file_arg_load_model_2 = "../../mpathic/MPAthic_tests/input/model_bad_mat_floatpos.txt"
    good_file_arg_load_model_1 = "../../mpathic/data/sortseq/full-0/crp_model.txt"

    # test file_arg for io.load_model
    test_parameter_values(func=mpathic.io.load_model, var_name='file_arg',
                          fail_list=[bad_file_arg_load_model_1, bad_file_arg_load_model_2],
                          success_list=[good_file_arg_load_model_1])

    #############################
    ## io.load_model tests end ##
    #############################


# functional tests for simulate library
def test_simulate_library():

    # test default parameters
    test_parameter_values(func=mpathic.SimulateLibrary)

    # test wtseq
    test_parameter_values(func=mpathic.SimulateLibrary, var_name='wtseq', fail_list=[3, 1.0, "XxX", False, ""],
                          success_list=["ATTCCGAGTA", "ATGTGTAGTCGTAG"])
    # test mutation rate
    test_parameter_values(func=mpathic.SimulateLibrary, var_name='mutrate', fail_list=[1.1, 2, -1, 0], success_list=[0.5, 0.1])

    # test numseq
    test_parameter_values(func=mpathic.SimulateLibrary, var_name='numseq', fail_list=['x', -1, 0, 0.5], success_list=[1, 2, 3, 100])

    # test dicttype
    #test_parameter_values(func=mpa.simulate_library_class(wtseq=wtseq_dna),var_name='dicttype',fail_list=['x',1,True],success_list=['dna','rna','protein'])
    test_parameter_values(func=mpathic.SimulateLibrary, var_name='dicttype',
                          fail_list=['x', 1, True], success_list=['dna','rna','protein'])

    # Note *** Need valid example of probarr to test ***
    # test probarr
    test_parameter_values(func=mpathic.SimulateLibrary, var_name='probarr', fail_list=[1, 1.0, "x", [1, 2, 3]], success_list=[None])

    # tags
    test_parameter_values(func=mpathic.SimulateLibrary, var_name='tags', fail_list=[None, -1, 3.9], success_list=[True, False])

    # tag_length
    test_parameter_values(func=mpathic.SimulateLibrary, var_name='tag_length', fail_list=[None, -1, 3.9],success_list=[3, 200])

# functional tests for simulate sort
def test_simulate_sort():

    # mpathic good model
    model_good_df = mpathic.io.load_model('../../mpathic/data/sortseq/full-0/crp_model.txt')

    dataset_bad_df_1 = mpathic.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_bad_badseqs.txt")
    dataset_bad_df_2 = mpathic.io.load_model('../../mpathic/data/sortseq/full-0/crp_model.txt')
    dataset_good_df_1 = mpathic.io.load_dataset("../../mpathic/data/sortseq/full-0/data_small.txt")
    dataset_good_df_2 = mpathic.io.load_dataset("../../mpathic/data/sortseq/full-150/data.txt")
    dataset_good_df_3 = mpathic.io.load_dataset("../../mpathic/data/sortseq/full-500/data.txt")
    dataset_good_df_4 = mpathic.io.load_dataset("../../mpathic/data/sortseq/full-wt/data.txt")

    # test input df
    test_parameter_values(mpathic.SimulateSort, var_name='df',
                          fail_list=
                          [
                              0,
                              dataset_bad_df_1,
                              dataset_bad_df_2
                          ],
                          success_list=
                          [dataset_good_df_1,
                           dataset_good_df_2,
                           dataset_good_df_3,
                           dataset_good_df_4
                          ],
                          mp=model_good_df)

    # test model dataframe
    ss_rnap_model = mpathic.io.load_model('../../mpathic/data/sortseq/rnap-wt/rnap_model.txt')
    test_parameter_values(func=mpathic.SimulateSort, var_name='mp', fail_list=[None, 'x'],
                          success_list=[ss_rnap_model, model_good_df], df=dataset_good_df_1)

    # test noise type
    test_parameter_values(func=mpathic.SimulateSort, var_name='noisetype', fail_list=[1, 2.1, 'x', 'LogNormal'],
                          success_list=['Normal', 'None'], df=dataset_good_df_1,
                          mp=model_good_df)

    # test nbins
    test_parameter_values(func=mpathic.SimulateSort, var_name='nbins', fail_list=['x', -1, 1.3, 1],
                          success_list=[2, 3, 10], df=dataset_good_df_1,
                          mp=model_good_df)


# functional tests for profile frequencies.
def test_profile_freq():

    # dataset_df tests

    # note that the bad_df_1 never gets loaded because qc,
    # so none gets passed into profile_freq, which fails as expected
    bad_df_1 = mpathic.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_bad_badseqs.txt")

    good_df_1 = mpathic.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_crp.txt")
    good_df_2 = mpathic.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_good_pro.txt")

    test_parameter_values(func=mpathic.ProfileFreq, var_name='dataset_df', fail_list=[3, 'x', None, bad_df_1],
                          success_list=[good_df_1, good_df_2])

    # bin tests
    test_parameter_values(func=mpathic.ProfileFreq, var_name='bin', fail_list=[-1, 'x', 1.2],
                          success_list=[2, 3], dataset_df=good_df_1)

    # start tests
    test_parameter_values(func=mpathic.ProfileFreq, var_name='start', fail_list=[0.1, 'x', 1.2, None],
                          success_list=[2, 3, 4, 10], dataset_df=good_df_1)

    # end tests
    test_parameter_values(func=mpathic.ProfileFreq, var_name='end', fail_list=[0.1, 'x', 1.2],
                          success_list=[2, 3, 4, 10], dataset_df=good_df_1)


# function tests for profile mutrate
def test_profile_mut():

    bad_df_1 = mpathic.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_bad_badseqs.txt")
    good_df_1 = mpathic.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_crp.txt")
    good_df_2 = mpathic.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_good_pro.txt")

    test_parameter_values(func=mpathic.ProfileMut, var_name='dataset_df', fail_list=[3, 'x', None, bad_df_1],
                          success_list=[good_df_1, good_df_2])

    # bin tests
    test_parameter_values(func=mpathic.ProfileMut, var_name='bin', fail_list=[-1, 'x', 1.2],
                          success_list=[2, 3], dataset_df=good_df_1)

    # start tests
    test_parameter_values(func=mpathic.ProfileMut, var_name='start', fail_list=[0.1, 'x', 1.2, None],
                          success_list=[2, 3, 4, 10], dataset_df=good_df_1)

    # end tests
    test_parameter_values(func=mpathic.ProfileMut, var_name='end', fail_list=[0.1, 'x', 1.2],
                          success_list=[2, 3, 4, 10], dataset_df=good_df_1)

    # err tests
    test_parameter_values(func=mpathic.ProfileMut, var_name='err', fail_list=[0.1, 'x', 1, 'True', None],
                          success_list=[True, False], dataset_df=good_df_1)


# function tests for profile mutrate
def test_profile_info():

    bad_df_1 = mpathic.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_bad_badseqs.txt")
    good_df_1 = mpathic.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_crp.txt")
    good_df_2 = mpathic.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_good_pro.txt")

    #test_parameter_values(func=mpathic.ProfileInfo, var_name='dataset_df', fail_list=[3, 'x', None, bad_df_1],success_list=[good_df_1, good_df_2])
    
    # err tests
    #test_parameter_values(func=mpathic.ProfileInfo, var_name='err', fail_list=[0.1, 'x', 1, 'True', None],success_list=[True, False], dataset_df=good_df_1)

    # method tests
    #test_parameter_values(func=mpathic.ProfileInfo, var_name='method', fail_list=[0.1, 'x', 1, 'True', None],success_list=['naive','tpm','nsb'], dataset_df=good_df_1)
    test_parameter_values(func=mpathic.ProfileInfo, var_name='method', fail_list=[0.1, 'x', 1, 'True', None],success_list=['naive', 'tpm'], dataset_df=good_df_1)

    # pseudocount tests
    test_parameter_values(func=mpathic.ProfileInfo, var_name='pseudocount', fail_list=['x', 1, 'True', None, -1.4],success_list=[0.1,1.5,9.3], dataset_df=good_df_1)

    # start tests
    test_parameter_values(func=mpathic.ProfileInfo, var_name='start', fail_list=[0.1, 'x', 1.2, None],success_list=[2, 3, 4, 10], dataset_df=good_df_1)

    # end tests
    test_parameter_values(func=mpathic.ProfileInfo, var_name='end', fail_list=[0.1, 'x', 1.2],success_list=[2, 3, 4, 10], dataset_df=good_df_1)

# functional tests for learn model
def test_learn_model():

    learn_model_good_df = mpathic.io.load_dataset("../../mpathic/data/sortseq/full-0/data.txt")
    small_dataset = mpathic.io.load_dataset("../../mpathic/data/sortseq/full-0/data_small.txt")

    test_parameter_values(func=mpathic.LearnModel, var_name='df', fail_list=[None, 2.4],
                          success_list=[learn_model_good_df, small_dataset], lm='ER')

    # the following is problematic: for lm = 'PR', the convex_optimization algorithm experies overflows (line 574)
    # for lm = 'LS', the are some convergence warnings.
    # for lm = 'IM', the test passes but takes ~ 2 hours to complete
    #test_parameter_values(func=mpathic.LearnModel, var_name='lm', fail_list=[None, 2.4, 'x'],success_list=['ER','LS','PR', 'IM'], df=small_dataset)
    test_parameter_values(func=mpathic.LearnModel, var_name='lm', fail_list=[None, 2.4, 'x'],success_list=['ER', 'LS', 'IM'], df=small_dataset)

    # test modeltype


def test_evaluate_model():

    # setup variables for evaluate model
    loader = mpathic.io
    dataset_df = loader.load_dataset(mpathic.__path__[0] + '/data/sortseq/full-0/library.txt')
    mp_df = loader.load_model(mpathic.__path__[0] + '/examples/true_model.txt')
    ss = mpathic.SimulateSort(df=dataset_df, mp=mp_df)
    temp_ss = ss.output_df

    temp_ss = ss.output_df
    # print(temp_ss.columns)
    cols = ['ct', 'ct_0', 'ct_1', 'ct_2', 'ct_3', 'seq']
    temp_ss = temp_ss[cols]

    # test dataset_df
    test_parameter_values(func=mpathic.EvaluateModel, var_name='dataset_df', fail_list=[None, 2.4,'x'],success_list=[temp_ss],model_df=mp_df)

    # test model_df
    test_parameter_values(func=mpathic.EvaluateModel, var_name='model_df', fail_list=[None, 2.4, 'x'],success_list=[mp_df], dataset_df=temp_ss)

    # test left
    test_parameter_values(func=mpathic.EvaluateModel, var_name='left', fail_list=[2.4, 'x',-1,4],success_list=[None,0], model_df=mp_df,dataset_df=temp_ss)

    # test right
    test_parameter_values(func=mpathic.EvaluateModel, var_name='right', fail_list=[2.4, 'x', -1, 4],success_list=[None,22], model_df=mp_df, dataset_df=temp_ss)


def test_scan_model():


    loader = mpathic.io
    mp_df = loader.load_model(mpathic.__path__[0] + '/examples/true_model.txt')
    fastafile = mpathic.__path__[0] + "/examples/genome_ecoli_1000lines.fa"
    contig = mpathic.io.load_contigs_from_fasta(fastafile, mp_df)

    # model_df
    test_parameter_values(func=mpathic.ScanModel, var_name='model_df',fail_list=[None,12,'x',True],success_list=[mp_df],contig_list=contig)


def test_predictive_info():
    loader = mpathic.io

    loader = mpathic.io
    dataset_df = loader.load_dataset(mpathic.__path__[0] + '/data/sortseq/full-0/library.txt')
    mp_df = loader.load_model(mpathic.__path__[0] + '/examples/true_model.txt')
    ss = mpathic.SimulateSort(df=dataset_df, mp=mp_df)
    temp_ss = ss.output_df

    temp_ss = ss.output_df
    # print(temp_ss.columns)
    cols = ['ct', 'ct_0', 'ct_1', 'ct_2', 'ct_3', 'seq']
    temp_ss = temp_ss[cols]

    # data_df
    test_parameter_values(func=mpathic.PredictiveInfo,var_name='data_df',fail_list=[None,12,'x',True],success_list=[temp_ss],model_df=mp_df, start=0)

    # model_df
    test_parameter_values(func=mpathic.PredictiveInfo, var_name='model_df', fail_list=[None, 12, 'x', True],success_list=[mp_df], data_df=temp_ss, start=0)

    # start tests
    test_parameter_values(func=mpathic.PredictiveInfo, var_name='start', fail_list=[-1, 0.1, 'x', 1.2, None],success_list=[0], data_df=temp_ss,model_df=mp_df)

    # end tests. Should the value 0 here end in success?
    test_parameter_values(func=mpathic.PredictiveInfo, var_name='end', fail_list=[0.1, 'x', 1.2],success_list=[None,0], data_df=temp_ss, model_df=mp_df,start=0)



# temporary method used only for debugging. Could be deleted
def run_single_test():
    pass

#run_single_test()

test_mpathic_io()
test_simulate_library()
test_simulate_sort()
test_profile_freq()
test_profile_mut()
test_profile_info()
test_learn_model()
test_evaluate_model()
test_scan_model()
test_predictive_info()


