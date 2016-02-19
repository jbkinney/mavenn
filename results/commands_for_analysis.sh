'''commands to run all analysis. WARNING, this will take ~ 30 hours total.
Commands should probably be run individually'''

'''MCMC analysis'''
#learn CRP models
sst learn_matrix -lm memsaver -i crp-wt_formatted.txt -o crp-wt_MCMC_crp -s 3 -e 25
sst learn_matrix -lm memsaver -i full-wt_formatted.txt -o full-wt_MCMC_crp -s 3 -e 25
sst learn_matrix -lm memsaver -i full-500_formatted.txt -o full-500_MCMC_crp -s 3 -e 25
sst learn_matrix -lm memsaver -i full-150_formatted.txt -o full-150_MCMC_crp -s 3 -e 25
sst learn_matrix -lm memsaver -i full-0_formatted.txt -o full-0_MCMC_crp -s 3 -e 25

#learn RNAP models
sst learn_matrix -lm memsaver -i rnap-wt_formatted.txt -o rnap-wt_MCMC_rnap -s 37 -e 71
sst learn_matrix -lm memsaver -i full-wt_formatted.txt -o full-wt_MCMC_rnap -s 37 -e 71
sst learn_matrix -lm memsaver -i full-500_formatted.txt -o full-500_MCMC_rnap -s 37 -e 71
sst learn_matrix -lm memsaver -i full-150_formatted.txt -o full-150_MCMC_rnap -s 37 -e 71
sst learn_matrix -lm memsaver -i full-0_formatted.txt -o full-0_MCMC_rnap -s 37 -e 71

#learn DMS models of the two data sets
sst learn_matrix -lm memsaver -i dms_1_formatted -o dms_1_MCMC
sst learn_matrix -lm memsaver -i dms_2_formatted -o dms_2_MCMC

#learn mpra models 
sst learn_matrix -lm memsaver -i CRE_100uM_ds_formatted -o CRE_100uM_ds_MCMC
sst learn_matrix -lm memsaver -i CRE_100uM_test_formatted -o CRE_100uM_test_MCMC

'''calculate the predictive information'''


