#!/bin/bash

# additive models
for dataset in tdp43 amyloid gb1
do
    qsub -N add_$dataset -o output_add_$dataset \
    -e log_add_$dataset additive_models_elzar.sh $dataset
done

# mpsa models
for model in additive neighbor pairwise blackbox
do 
    qsub -N mpsa_$model -o output_mpsa_$model \
     -e log_mpsa_$model mpsa_models_elzar.sh $model
done

# sortseq model
qsub -N sortseq -o output_sortseq -e log_sortseq sortseq_thermo_elzar.sh

# gb1 thermodynamic model with arg_1 = number of epochs and arg_2 = learning_rate
qsub -N thermo_gb1 -o output_thermo_gb1 -e log_themro_gb1 gb1_thermo_elzar.sh 3000 2e-3

# figs1 models
qsub -N figs1 -o output_figs1 -e log_figs1 figs1_models_elzar.sh 2

# gb1 balckbox
qsub -N gb1_bb -o output_gb1_bb -e log_gb1_bb gb1_blackbox_elzar.sh blackbox