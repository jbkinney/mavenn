#!/bin/bash
#$ -cwd
#$ -l m_mem_free=8G
#$ -pe threads 8
python figs1_models_training.py $1