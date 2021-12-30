#!/bin/bash
#$ -cwd
#$ -l m_mem_free=16G
#$ -pe threads 8
python gb1_thermo_training.py -e $1 -lr $2