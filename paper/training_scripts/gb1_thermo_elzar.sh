#!/bin/bash
#$ -cwd
#$ -l m_mem_free=16G
#$ -pe threads 8
#$ -o output_2e-3
#$ -e log_2e-3
python gb1_thermo_training.py -e 3000 -lr 2e-3
