#!/bin/bash
#$ -cwd
#$ -l m_mem_free=16G
#$ -pe threads 8
#$ -o output_mpsa_pairwise
#$ -e log_mpsa_pairwise
python mpsa_models_training.py -m pairwise