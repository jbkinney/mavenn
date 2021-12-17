#!/bin/bash
#$ -cwd
#$ -l m_mem_free=16G
#$ -pe threads 16
#$ -o output_5e-4
#$ -e log_5e-4
python gb1_thermo_training.py -e 4000 -lr 5e-4
