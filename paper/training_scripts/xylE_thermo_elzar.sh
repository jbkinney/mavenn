#!/bin/bash
#$ -cwd
#$ -l m_mem_free=32G
#$ -pe threads 32

python xylE_thermo_training.py -ep $1 -lr $2 -bs $3 -re $4