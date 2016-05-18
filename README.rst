SortSeqTools is a software package for use in analyzing Sort-Seq,
Massively Parallel Reporter Assays, Deep Protein Mutagenesis Assays, and Selex-Seq
experiments.

For help using and installing this package please refer to our paper, the
github examples, and the `sortseq_tools documentation`_

.. _`sortseq_tools documentation`: http://jbkinney.github.io/sortseq

Code for Analysis:

The commands for producing the models analysed in our paper are below.
In order to run these commands download the code from github, open a terminal, and navigate to the results
folder. Copy one of the following commands into the terminal and it will produce one model.
To run all the commands will take approximately 30 hours on a laptop.

CRP matrix models:
sortseq_tools learn_model -lm IM -i crp_models/crp-wt_formatted.txt -o crp-wt_MCMC_crp -s 3 -e 25
sortseq_tools learn_model -lm LS -i crp_models/crp-wt_formatted.txt -o crp-wt_LS_crp -s 3 -e 25
sortseq_tools learn_model -lm ER -i crp_models/crp-wt_formatted.txt -o crp-wt_ER_crp -s 3 -e 25
sortseq_tools learn_model -lm IM -i crp_models/full-wt_formatted.txt -o full-wt_MCMC_crp -s 3 -e 25
sortseq_tools learn_model -lm LS -i crp_models/full-wt_formatted.txt -o full-wt_LS_crp -s 3 -e 25
sortseq_tools learn_model -lm ER -i crp_models/full-wt_formatted.txt -o full-wt_ER_crp -s 3 -e 25
sortseq_tools learn_model -lm IM -i crp_models/full-500_formatted.txt -o full-500_MCMC_crp -s 3 -e 25
sortseq_tools learn_model -lm LS -i crp_models/full-500_formatted.txt -o full-500_LS_crp -s 3 -e 25
sortseq_tools learn_model -lm ER -i crp_models/full-500_formatted.txt -o full-500_ER_crp -s 3 -e 25
sortseq_tools learn_model -lm IM -i crp_models/full-150_formatted.txt -o full-150_MCMC_crp -s 3 -e 25
sortseq_tools learn_model -lm LS -i crp_models/full-150_formatted.txt -o full-150_LS_crp -s 3 -e 25
sortseq_tools learn_model -lm ER -i crp_models/full-150_formatted.txt -o full-150_ER_crp -s 3 -e 25
sortseq_tools learn_model -lm IM -i crp_models/full-0_formatted.txt -o full-0_MCMC_crp -s 3 -e 25
sortseq_tools learn_model -lm LS -i crp_models/full-0_formatted.txt -o full-0_LS_crp -s 3 -e 25
sortseq_tools learn_model -lm ER -i crp_models/full-0_formatted.txt -o full-0_ER_crp -s 3 -e 25

CRP neighbor models:
sortseq_tools learn_model -lm IM -i crp_models/crp-wt_formatted.txt -o crp-wt_MCMC_crp_NBR -s 3 -e 25 -mt NBR
sortseq_tools learn_model -lm IM -i crp_models/full-wt_formatted.txt -o full-wt_MCMC_crp_NBR -s 3 -e 25 -mt NBR
sortseq_tools learn_model -lm IM -i crp_models/full-500_formatted.txt -o full-500_MCMC_crp_NBR -s 3 -e 25 -mt NBR
sortseq_tools learn_model -lm IM -i crp_models/full-150_formatted.txt -o full-150_MCMC_crp_NBR -s 3 -e 25 -mt NBR
sortseq_tools learn_model -lm IM -i crp_models/full-0_formatted.txt -o full-0_MCMC_crp_NBR -s 3 -e 25 -mt NBR

RNAP matrix models:

sortseq_tools learn_model -lm IM -i rnap_models/rnap-wt_formatted.txt -o rnap-wt_MCMC_rnap -s 37 -e 71

sortseq_tools learn_model -lm LS -i rnap_models/rnap-wt_formatted.txt -o rnap-wt_LS_rnap -s 37 -e 71

sortseq_tools learn_model -lm ER -i rnap_models/rnap-wt_formatted.txt -o rnap-wt_ER_rnap -s 37 -e 71

sortseq_tools learn_model -lm IM -i rnap_models/full-wt_formatted.txt -o full-wt_MCMC_rnap -s 37 -e 71

sortseq_tools learn_model -lm LS -i rnap_models/full-wt_formatted.txt -o full-wt_LS_rnap -s 37 -e 71

sortseq_tools learn_model -lm ER -i rnap_models/full-wt_formatted.txt -o full-wt_ER_rnap -s 37 -e 71

sortseq_tools learn_model -lm IM -i rnap_models/full-500_formatted.txt -o full-500_MCMC_rnap -s 37 -e 71
sortseq_tools learn_model -lm LS -i rnap_models/full-500_formatted.txt -o full-500_LS_rnap -s 37 -e 71
sortseq_tools learn_model -lm ER -i rnap_models/full-500_formatted.txt -o full-500_ER_rnap -s 37 -e 71
sortseq_tools learn_model -lm IM -i rnap_models/full-150_formatted.txt -o full-150_MCMC_rnap -s 37 -e 71
sortseq_tools learn_model -lm LS -i rnap_models/full-150_formatted.txt -o full-150_LS_rnap -s 37 -e 71
sortseq_tools learn_model -lm ER -i rnap_models/full-150_formatted.txt -o full-150_ER_rnap -s 37 -e 71
sortseq_tools learn_model -lm IM -i rnap_models/full-0_formatted.txt -o full-0_MCMC_rnap -s 37 -e 71
sortseq_tools learn_model -lm LS -i rnap_models/full-0_formatted.txt -o full-0_LS_rnap -s 37 -e 71
sortseq_tools learn_model -lm ER -i rnap_models/full-0_formatted.txt -o full-0_ER_rnap -s 37 -e 71

#learn RNAP neighbor models
sortseq_tools learn_model -lm IM -i rnap_models/crp-wt_formatted.txt -o crp-wt_MCMC_crp_NBR -s 37 -e 71 -mt NBR
sortseq_tools learn_model -lm IM -i rnap_models/full-wt_formatted.txt -o full-wt_MCMC_crp_NBR -s 37 -e 71 -mt NBR
sortseq_tools learn_model -lm IM -i rnap_models/full-500_formatted.txt -o full-500_MCMC_crp_NBR -s 37 -e 71 -mt NBR
sortseq_tools learn_model -lm IM -i rnap_models/full-150_formatted.txt -o full-150_MCMC_crp_NBR -s 37 -e 71 -mt NBR
sortseq_tools learn_model -lm IM -i rnap_models/full-0_formatted.txt -o full-0_MCMC_crp_NBR -s 37 -e 71 -mt NBR

#learn DMS models of the two data sets
sortseq_tools learn_model -lm IM -i dms/dms_1_formatted -o dms_1_MCMC
sortseq_tools learn_model -lm LS -i dms/dms_1_formatted -o dms_1_LS
sortseq_tools learn_model -lm ER -i dms/dms_1_formatted -o dms_1_ER
sortseq_tools learn_model -lm IM -i dms/dms_2_formatted -o dms_2_MCMC
sortseq_tools learn_model -lm LS -i dms/dms_2_formatted -o dms_2_LS
sortseq_tools learn_model -lm ER -i dms/dms_2_formatted -o dms_2_ER

#learn mpra models 
sortseq_tools learn_model -lm IM -i mpra/CRE_100uM_ds_formatted -o CRE_100uM_ds_MCMC
sortseq_tools learn_model -lm LS -i mpra/CRE_100uM_ds_formatted -o CRE_100uM_ds_LS
sortseq_tools learn_model -lm ER -i mpra/CRE_100uM_ds_formatted -o CRE_100uM_ds_ER
sortseq_tools learn_model -lm IM -i mpra/CRE_100uM_test_formatted -o CRE_100uM_test_MCMC
sortseq_tools learn_model -lm LS -i mpra/CRE_100uM_test_formatted -o CRE_100uM_test_LS
sortseq_tools learn_model -lm ER -i mpra/CRE_100uM_test_formatted -o CRE_100uM_test_ER

#learn mpra Neighbor models
sortseq_tools learn_model -lm IM -i mpra/CRE_100uM_ds_formatted -o CRE_100uM_ds_MCMC_NBR -mt NBR
sortseq_tools learn_model -lm IM -i mpra/CRE_100uM_test_formatted -o CRE_100uM_test_MCMC_NBR -mt NBR
