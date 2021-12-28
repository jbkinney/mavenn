#!/bin/bash
#$ -cwd
#$ -l m_mem_free=8G
#$ -pe threads 8
#$ -o output_add_tdp43
#$ -e log_add_tdp43
python additive_models_training.py -d tdp43
