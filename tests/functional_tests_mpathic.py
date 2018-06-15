from __future__ import print_function   # so that print behaves like python 3.x not a special lambda statement
import mpathic as mpa
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
            func=mpa.io.load_dataset, var_name='file_arg',
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
    test_parameter_values(func=mpa.io.load_dataset, var_name='file_arg',
                          fail_list=[bad_file_arg_load_dataset_fasta_1],
                          success_list=[good_file_arg_load_dataset_fasta_1, good_file_arg_load_dataset_fasta_2],
                          file_type='fasta', seq_type='dna')

    # test parameter file args for load_dataset for file_type = fastq, seq_type = 'dna'

    bad_file_arg_load_dataset_fastq_2 = "../../mpathic/MPAthic_tests/input/seq_bad_hasNs.fastq"
    good_file_arg_load_dataset_fastq_1 = "../../mpathic/MPAthic_tests/input/seq_good.fastq"

    # this file causes trouble: succeeds but should fail?
    bad_file_arg_load_dataset_fastq_1 = "../../mpathic/MPAthic_tests/input/seq_bad_actuallyfasta.fastq"

    # test parameter file args for load_dataset for file_type = fasta, seq_type = 'dna'
    test_parameter_values(func=mpa.io.load_dataset, var_name='file_arg',
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
    test_parameter_values(func=mpa.io.load_model, var_name='file_arg',
                          fail_list=[bad_file_arg_load_model_1, bad_file_arg_load_model_2],
                          success_list=[good_file_arg_load_model_1])

    #############################
    ## io.load_model tests end ##
    #############################


# functional tests for simulate library
def test_simulate_library():

    # test default parameters
    test_parameter_values(func=mpa.SimulateLibrary)

    # test wtseq
    test_parameter_values(func=mpa.SimulateLibrary, var_name='wtseq', fail_list=[3, 1.0, "XxX", False, ""],
                          success_list=["ATTCCGAGTA", "ATGTGTAGTCGTAG"])
    # test mutation rate
    test_parameter_values(func=mpa.SimulateLibrary, var_name='mutrate', fail_list=[1.1, 2, -1, 0], success_list=[0.5, 0.1])

    # test numseq
    test_parameter_values(func=mpa.SimulateLibrary, var_name='numseq', fail_list=['x', -1, 0, 0.5], success_list=[1, 2, 3, 100])

    # test dicttype
    #test_parameter_values(func=mpa.simulate_library_class(wtseq=wtseq_dna),var_name='dicttype',fail_list=['x',1,True],success_list=['dna','rna','protein'])
    test_parameter_values(func=mpa.SimulateLibrary, var_name='dicttype',
                          fail_list=['x', 1, True], success_list=['dna','rna','protein'])

    # Note *** Need valid example of probarr to test ***
    # test probarr
    test_parameter_values(func=mpa.SimulateLibrary, var_name='probarr', fail_list=[1, 1.0, "x", [1, 2, 3]], success_list=[None])

    # tags
    test_parameter_values(func=mpa.SimulateLibrary, var_name='tags', fail_list=[None, -1, 3.9], success_list=[True, False])

    # tag_length
    test_parameter_values(func=mpa.SimulateLibrary, var_name='tag_length', fail_list=[None, -1, 3.9],
                          success_list=[3, 200])

# functional tests for simulate sort
def test_simulate_sort():

    # mpathic good model
    model_good_df = mpa.io.load_model('../../mpathic/data/sortseq/full-0/crp_model.txt')

    dataset_bad_df_1 = mpa.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_bad_badseqs.txt")
    dataset_bad_df_2 = mpa.io.load_model('../../mpathic/data/sortseq/full-0/crp_model.txt')
    dataset_good_df_1 = mpa.io.load_dataset("../../mpathic/data/sortseq/full-0/data_small.txt")
    dataset_good_df_2 = mpa.io.load_dataset("../../mpathic/data/sortseq/full-150/data.txt")
    dataset_good_df_3 = mpa.io.load_dataset("../../mpathic/data/sortseq/full-500/data.txt")
    dataset_good_df_4 = mpa.io.load_dataset("../../mpathic/data/sortseq/full-wt/data.txt")

    # test input df
    test_parameter_values(mpa.SimulateSort, var_name='df',
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
    ss_rnap_model = mpa.io.load_model('../../mpathic/data/sortseq/rnap-wt/rnap_model.txt')
    test_parameter_values(func=mpa.SimulateSort, var_name='mp', fail_list=[None, 'x'],
                          success_list=[ss_rnap_model, model_good_df], df=dataset_good_df_1)

    # test noise type
    test_parameter_values(func=mpa.SimulateSort, var_name='noisetype', fail_list=[1, 2.1, 'x', 'LogNormal'],
                          success_list=['Normal', 'None'], df=dataset_good_df_1,
                          mp=model_good_df)

    # test nbins
    test_parameter_values(func=mpa.SimulateSort, var_name='nbins', fail_list=['x', -1, 1.3, 1],
                          success_list=[2, 3, 10], df=dataset_good_df_1,
                          mp=model_good_df)


# functional tests for profile frequencies.
def test_profile_freq():

    # dataset_df tests

    # note that the bad_df_1 never gets loaded because qc,
    # so none gets passed into profile_freq, which fails as expected
    bad_df_1 = mpa.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_bad_badseqs.txt")

    good_df_1 = mpa.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_crp.txt")
    good_df_2 = mpa.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_good_pro.txt")

    test_parameter_values(func=mpa.ProfileFreq, var_name='dataset_df', fail_list=[3, 'x', None, bad_df_1],
                          success_list=[good_df_1, good_df_2])

    # bin tests
    test_parameter_values(func=mpa.ProfileFreq, var_name='bin', fail_list=[-1, 'x', 1.2],
                          success_list=[2, 3], dataset_df=good_df_1)

    # start tests
    test_parameter_values(func=mpa.ProfileFreq, var_name='start', fail_list=[0.1, 'x', 1.2, None],
                          success_list=[2, 3, 4, 10], dataset_df=good_df_1)

    # end tests
    test_parameter_values(func=mpa.ProfileFreq, var_name='end', fail_list=[0.1, 'x', 1.2],
                          success_list=[2, 3, 4, 10], dataset_df=good_df_1)


# function tests for profile mutrate
def test_profile_mut():

    bad_df_1 = mpa.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_bad_badseqs.txt")
    good_df_1 = mpa.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_crp.txt")
    good_df_2 = mpa.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_good_pro.txt")

    test_parameter_values(func=mpa.ProfileMut, var_name='dataset_df', fail_list=[3, 'x', None, bad_df_1],
                          success_list=[good_df_1, good_df_2])

    # bin tests
    test_parameter_values(func=mpa.ProfileMut, var_name='bin', fail_list=[-1, 'x', 1.2],
                          success_list=[2, 3], dataset_df=good_df_1)

    # start tests
    test_parameter_values(func=mpa.ProfileMut, var_name='start', fail_list=[0.1, 'x', 1.2, None],
                          success_list=[2, 3, 4, 10], dataset_df=good_df_1)

    # end tests
    test_parameter_values(func=mpa.ProfileMut, var_name='end', fail_list=[0.1, 'x', 1.2],
                          success_list=[2, 3, 4, 10], dataset_df=good_df_1)

    # err tests
    test_parameter_values(func=mpa.ProfileMut, var_name='err', fail_list=[0.1, 'x', 1, 'True', None],
                          success_list=[True, False], dataset_df=good_df_1)


# function tests for profile mutrate
def test_profile_info():

    bad_df_1 = mpa.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_bad_badseqs.txt")
    good_df_1 = mpa.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_crp.txt")
    good_df_2 = mpa.io.load_dataset("../../mpathic/MPAthic_tests/input/dataset_good_pro.txt")

    test_parameter_values(func=mpa.ProfileInfo, var_name='dataset_df', fail_list=[3, 'x', None, bad_df_1],
                          success_list=[good_df_1, good_df_2])
    
    # err tests
    test_parameter_values(func=mpa.ProfileInfo, var_name='err', fail_list=[0.1, 'x', 1, 'True', None],
                          success_list=[True, False], dataset_df=good_df_1)

    # method tests
    test_parameter_values(func=mpa.ProfileInfo, var_name='method', fail_list=[0.1, 'x', 1, 'True', None],
                          success_list=['naive','tpm','nsb'], dataset_df=good_df_1)

    # pseudocount tests
    test_parameter_values(func=mpa.ProfileInfo, var_name='pseudocount', fail_list=['x', 1, 'True', None, -1.4],
                          success_list=[0.1,1.5,9.3], dataset_df=good_df_1)

    # start tests
    test_parameter_values(func=mpa.ProfileInfo, var_name='start', fail_list=[0.1, 'x', 1.2, None],
                          success_list=[2, 3, 4, 10], dataset_df=good_df_1)

    # end tests
    test_parameter_values(func=mpa.ProfileInfo, var_name='end', fail_list=[0.1, 'x', 1.2],
                          success_list=[2, 3, 4, 10], dataset_df=good_df_1)

# functional tests for learn model
def test_learn_model():

    learn_model_good_df = mpa.io.load_dataset("../../mpathic/data/sortseq/full-0/data.txt")
    small_dataset = mpa.io.load_dataset("../../mpathic/data/sortseq/full-0/data_small.txt")

    test_parameter_values(func=mpa.LearnModel, var_name='df', fail_list=[None, 2.4],
                          success_list=[learn_model_good_df, small_dataset], lm='ER')

    # the following is problematic: for lm = 'PR', the convex_optimization algorithm experies overflows (line 574)
    # for lm = 'LS', the are some convergence warnings.
    # for lm = 'IM', the test passes but takes ~ 2 hours to complete
    test_parameter_values(func=mpa.LearnModel, var_name='lm', fail_list=[None, 2.4, 'x'],
                          success_list=['ER','LS','PR', 'IM'], df=small_dataset)

    # test modeltype


# temporary method used only for debugging. Will be deleted in production.
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
