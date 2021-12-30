#!/bin/bash
#$ -cwd
#$ -l m_mem_free=16G
#$ -pe threads 8
python mpsa_models_training.py -m $1