#!/bin/bash
#$ -cwd
#$ -l m_mem_free=16G
#$ -pe threads 16
python gb1_blackbox_training.py -m $1