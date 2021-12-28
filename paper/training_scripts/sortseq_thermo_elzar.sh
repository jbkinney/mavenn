#!/bin/bash
#$ -cwd
#$ -l m_mem_free=16G
#$ -pe threads 8
#$ -o output_sortseq
#$ -e log_sortseq
python sortseq_thermo_training.py 