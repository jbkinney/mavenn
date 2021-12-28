#!/bin/bash
#$ -cwd
#$ -l m_mem_free=8G
#$ -pe threads 8
#$ -o output_add_gb1
#$ -e log_add_gb1
python additive_models_training.py -d gb1
