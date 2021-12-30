#!/bin/bash
#$ -cwd
#$ -l m_mem_free=8G
#$ -pe threads 8
python additive_models_training.py -d $1