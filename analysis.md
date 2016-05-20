Analysis commands

The commands needed for reproducing the results described in Ireland & Kinney (2016) are provided below. To run any of these commands, navigate to the folder 'data/formatted/' and execute the command shown at the command line. 
Most of the commands below are for quantative model inference. To compute the predictive information values of a given model model.txt on a dataset dataset.txt, the following command is used

sortseq_tools predictiveinfo -ds dataset.txt -m model.txt -o outfile.txt

For example,

sortseq_tools predictiveinfo -ds sortseq/crp-wt-formatted.txt -m crp-wt_LS_crp.txt -o outfile.txt -s 3 -e 25

Please note that running all of the commands below on a standard laptop computer takes approximately 30 hours. 

Commands for Figure 4:

CRP matrix models:

sortseq_tools learn_model -lm IM -i sortseq/crp-wt-formatted.txt -o crp-wt_MCMC_crp.txt -s 3 -e 25

sortseq_tools learn_model -lm LS -i sortseq/crp-wt-formatted.txt -o crp-wt_LS_crp.txt -s 3 -e 25

sortseq_tools learn_model -lm ER -i sortseq/crp-wt-formatted.txt -o crp-wt_ER_crp.txt -s 3 -e 25

sortseq_tools learn_model -lm IM -i sortseq/full-wt-formatted.txt -o full-wt_MCMC_crp.txt -s 3 -e 25

sortseq_tools learn_model -lm LS -i sortseq/full-wt-formatted.txt -o full-wt_LS_crp.txt -s 3 -e 25

sortseq_tools learn_model -lm ER -i sortseq/full-wt-formatted.txt -o full-wt_ER_crp.txt -s 3 -e 25

sortseq_tools learn_model -lm IM -i sortseq/full-500-formatted.txt -o full-500_MCMC_crp.txt -s 3 -e 25

sortseq_tools learn_model -lm LS -i sortseq/full-500-formatted.txt -o full-500_LS_crp.txt -s 3 -e 25

sortseq_tools learn_model -lm ER -i sortseq/full-500-formatted.txt -o full-500_ER_crp.txt -s 3 -e 25

sortseq_tools learn_model -lm IM -i sortseq/full-150-formatted.txt -o full-150_MCMC_crp.txt -s 3 -e 25

sortseq_tools learn_model -lm LS -i sortseq/full-150-formatted.txt -o full-150_LS_crp.txt -s 3 -e 25

sortseq_tools learn_model -lm ER -i sortseq/full-150-formatted.txt -o full-150_ER_crp.txt -s 3 -e 25

sortseq_tools learn_model -lm IM -i sortseq/full-0-formatted.txt -o full-0_MCMC_crp.txt -s 3 -e 25

sortseq_tools learn_model -lm LS -i sortseq/full-0-formatted.txt -o full-0_LS_crp.txt -s 3 -e 25

sortseq_tools learn_model -lm ER -i sortseq/full-0-formatted.txt -o full-0_ER_crp.txt -s 3 -e 25

CRP neighbor models:

sortseq_tools learn_model -lm IM -i sortseq/crp-wt-formatted.txt -o crp-wt_MCMC_crp_NBR.txt -s 3 -e 25 -mt NBR

sortseq_tools learn_model -lm IM -i sortseq/full-wt-formatted.txt -o full-wt_MCMC_crp_NBR.txt -s 3 -e 25 -mt NBR

sortseq_tools learn_model -lm IM -i sortseq/full-500-formatted.txt -o full-500_MCMC_crp_NBR.txt -s 3 -e 25 -mt NBR

sortseq_tools learn_model -lm IM -i sortseq/full-150-formatted.txt -o full-150_MCMC_crp_NBR.txt -s 3 -e 25 -mt NBR

sortseq_tools learn_model -lm IM -i sortseq/full-0-formatted.txt -o full-0_MCMC_crp_NBR.txt -s 3 -e 25 -mt NBR

RNAP matrix models:

sortseq_tools learn_model -lm IM -i sortseq/rnap-wt-formatted.txt -o rnap-wt_MCMC_rnap.txt -s 37 -e 71

sortseq_tools learn_model -lm LS -i sortseq/rnap-wt-formatted.txt -o rnap-wt_LS_rnap.txt -s 37 -e 71

sortseq_tools learn_model -lm ER -i sortseq/rnap-wt-formatted.txt -o rnap-wt_ER_rnap.txt -s 37 -e 71

sortseq_tools learn_model -lm IM -i sortseq/full-wt-formatted.txt -o full-wt_MCMC_rnap.txt -s 37 -e 71

sortseq_tools learn_model -lm LS -i sortseq/full-wt-formatted.txt -o full-wt_LS_rnap.txt -s 37 -e 71

sortseq_tools learn_model -lm ER -i sortseq/full-wt-formatted.txt -o full-wt_ER_rnap.txt -s 37 -e 71

sortseq_tools learn_model -lm IM -i sortseq/full-500-formatted.txt -o full-500_MCMC_rnap.txt -s 37 -e 71

sortseq_tools learn_model -lm LS -i sortseq/full-500-formatted.txt -o full-500_LS_rnap.txt -s 37 -e 71

sortseq_tools learn_model -lm ER -i sortseq/full-500-formatted.txt -o full-500_ER_rnap.txt -s 37 -e 71

sortseq_tools learn_model -lm IM -i sortseq/full-150-formatted.txt -o full-150_MCMC_rnap.txt -s 37 -e 71

sortseq_tools learn_model -lm LS -i sortseq/full-150-formatted.txt -o full-150_LS_rnap.txt -s 37 -e 71

sortseq_tools learn_model -lm ER -i sortseq/full-150-formatted.txt -o full-150_ER_rnap.txt -s 37 -e 71

sortseq_tools learn_model -lm IM -i sortseq/full-0-formatted.txt -o full-0_MCMC_rnap.txt -s 37 -e 71

sortseq_tools learn_model -lm LS -i sortseq/full-0-formatted.txt -o full-0_LS_rnap.txt -s 37 -e 71

sortseq_tools learn_model -lm ER -i sortseq/full-0-formatted.txt -o full-0_ER_rnap.txt -s 37 -e 71

RNAP neighbor models:

sortseq_tools learn_model -lm IM -i sortseq/crp-wt-formatted.txt -o crp-wt_MCMC_crp_NBR.txt -s 37 -e 71 -mt NBR

sortseq_tools learn_model -lm IM -i sortseq/full-wt-formatted.txt -o full-wt_MCMC_crp_NBR.txt -s 37 -e 71 -mt NBR

sortseq_tools learn_model -lm IM -i sortseq/full-500-formatted.txt -o full-500_MCMC_crp_NBR.txt -s 37 -e 71 -mt NBR

sortseq_tools learn_model -lm IM -i sortseq/full-150-formatted.txt -o full-150_MCMC_crp_NBR.txt -s 37 -e 71 -mt NBR

sortseq_tools learn_model -lm IM -i sortseq/full-0-formatted.txt -o full-0_MCMC_crp_NBR.txt -s 37 -e 71 -mt NBR

Commands for Figure 5:

CRP dataset simulation:

sortseq_tools simulate_library  -w ACAGGTATACAGTACCATACCT -n 1000000 -o Simulated/CRP/simulated_library_train.txt

sortseq_tools simulate_sort -m Simulated/CRP/true_model.txt -n_bins 2 -i simulated_library_train.txt -o Simulated/CRP/2_bin_sorted-formatted.txt

sortseq_tools simulate_sort -m Simulated/CRP/true_model.txt -n_bins 10 -i simulated_library_train.txt -o Simulated/CRP/10_bin_sorted-formatted.txt

sortseq_tools simulate_library  -w ACAGGTATACAGTACCATACCT -n 1000000 -o Simulated/CRP/simulated_library_test.txt

sortseq_tools simulate_sort -m Simulated/CRP/true_model.txt -n_bins 2 -i simulated_library_test.txt -o Simulated/CRP/2_bin_sorted_test-formatted.txt

sortseq_tools simulate_sort -m Simulated/CRP/true_model.txt -n_bins 10 -i simulated_library_test.txt -o Simulated/CRP/10_bin_sorted_test-formatted.txt

CRP model inference:

sortseq_tools learn_model -lm IM -i Simulated/CRP/2_bin_sorted-formatted.txt -o sim_2_bin_IM_crp.txt 

sortseq_tools learn_model -lm LS -i Simulated/CRP/2_bin_sorted-formatted.txt -o sim_2_bin_LS_crp.txt 

sortseq_tools learn_model -lm ER -i Simulated/CRP/2_bin_sorted-formatted.txt -o sim_2_bin_ER_crp.txt

sortseq_tools learn_model -lm IM -i Simulated/CRP/2_bin_sorted-formatted.txt -o sim_2_bin_IM_crp_NBR.txt -mt NBR

sortseq_tools learn_model -lm LS -i Simulated/CRP/2_bin_sorted-formatted.txt -o sim_2_bin_LS_crp_NBR.txt -mt NBR

RNAP dataset simulation:

sortseq_tools simulate_library  -w GCTTTACACTTTATGCTTCCGGCTCGTATGTTGT -n 1000000 -o Simulated/RNAP/simulated_library_train.txt

sortseq_tools simulate_sort -m Simulated/RNAP/true_model.txt -n_bins 2 -i simulated_library_train.txt -o Simulated/RNAP/RNAP_2_bin_sorted-formatted.txt

sortseq_tools simulate_sort -m Simulated/RNAP/true_model.txt -n_bins 10 -i simulated_library_train.txt -o Simulated/RNAP/RNAP_10_bin_sorted-formatted.txt

sortseq_tools simulate_library  -w GCTTTACACTTTATGCTTCCGGCTCGTATGTTGT -n 1000000 -o Simulated/RNAP/simulated_library_test.txt

sortseq_tools simulate_sort -m Simulated/RNAP/true_model.txt -n_bins 2 -i simulated_library_test.txt -o Simulated/RNAP/RNAP_2_bin_sorted_test-formatted.txt

sortseq_tools simulate_sort -m Simulated/RNAP/true_model.txt -n_bins 10 -i simulated_library_test.txt -o Simulated/RNAP/RNAP_10_bin_sorted_test-formatted.txt

RNAP dataset inference:

sortseq_tools learn_model -lm IM -i Simulated/RNAP/RNAP_2_bin_sorted-formatted.txt -o sim_2_bin_IM_RNAP.txt 

sortseq_tools learn_model -lm LS -i Simulated/RNAP/RNAP_2_bin_sorted-formatted.txt -o sim_2_bin_LS_RNAP.txt 

sortseq_tools learn_model -lm ER -i Simulated/RNAP/RNAP_2_bin_sorted-formatted.txt -o sim_2_bin_ER_RNAP.txt

sortseq_tools learn_model -lm IM -i Simulated/RNAP/RNAP_2_bin_sorted-formatted.txt -o sim_2_bin_IM_RNAP_NBR.txt -mt NBR

sortseq_tools learn_model -lm LS -i Simulated/RNAP/RNAP_2_bin_sorted-formatted.txt -o sim_2_bin_LS_RNAP_NBR.txt -mt NBR

Commands for Figure 6:

DMS matrix models:

sortseq_tools learn_model -lm IM -i dms/dms_1_formatted.txt -o dms_1_MCMC.txt

sortseq_tools learn_model -lm LS -i dms/dms_1_formatted.txt -o dms_1_LS.txt

sortseq_tools learn_model -lm ER -i dms/dms_1_formatted.txt -o dms_1_ER.txt

sortseq_tools learn_model -lm IM -i dms/dms_2_formatted.txt -o dms_2_MCMC.txt

sortseq_tools learn_model -lm LS -i dms/dms_2_formatted.txt -o dms_2_LS.txt

sortseq_tools learn_model -lm ER -i dms/dms_2_formatted.txt -o dms_2_ER.txt

MPRA matrix models: 

sortseq_tools learn_model -lm IM -i mpra/CRE_100uM_ds_formatted.txt -o CRE_100uM_ds_MCMC.txt

sortseq_tools learn_model -lm LS -i mpra/CRE_100uM_ds_formatted.txt -o CRE_100uM_ds_LS.txt

sortseq_tools learn_model -lm ER -i mpra/CRE_100uM_ds_formatted.txt -o CRE_100uM_ds_ER.txt

sortseq_tools learn_model -lm IM -i mpra/CRE_100uM_test_formatted.txt -o CRE_100uM_test_MCMC.txt

sortseq_tools learn_model -lm LS -i mpra/CRE_100uM_test_formatted.txt -o CRE_100uM_test_LS.txt

sortseq_tools learn_model -lm ER -i mpra/CRE_100uM_test_formatted.txt -o CRE_100uM_test_ER.txt

