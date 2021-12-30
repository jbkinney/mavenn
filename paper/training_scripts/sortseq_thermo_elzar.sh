#!/bin/bash
#$ -cwd
#$ -l m_mem_free=16G
#$ -pe threads 8
python sortseq_thermo_training.py 