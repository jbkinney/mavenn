#!/bin/bash
#$ -cwd
#$ -l m_mem_free=16G
#$ -pe threads 8
#$ -o output_mpsa_blackbox
#$ -e log_mpsa_blackbox
python mpsa_models_training.py -m blackbox