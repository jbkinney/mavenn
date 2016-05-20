# Sort-Seq Tools 
version 0.01.01
Written by William T. Ireland and Justin B. Kinney
Copyright 2016
========

Overview

Sort-Seq Tools is a software package for analyzing data from a variety massively parallel assays, including Sort-Seq assays, Massively Parallel Reporter assays, and Deep Mutational Scanning assays. Sort-Seq Tools provides a set of command line routines, which are listed in the documentation [link]. Details can be found in the accompanying paper

Ireland WT, Kinney JB (2016) Sort-Seq Tools: modeling sequence-function relationships from massively parallel assays. bioRxiv doi:???

Requriements

Sort-Seq Tools is written in Python 2.9.7 and in Cython 0.23.4. It requires that the following Python packages also be installed: biopython, pymc, scikit-learn, statsmodels, mpmath, pandas, weblogo, and matplotlib. Sort-Seq Tools is currently in alpha testing and has been verified to work on Linux and Mac OS X. 

Installation

To install Sort-Seq Tools, clone this repository, navigate to the folder containing this README file, and execute

python setup.py install

Alternatively, Sort-Seq Tools can be installed from PyPI by executing

pip install sortseq_tools

This approach will also install all of Sort-Seq Tools's dependencies. After Sort-Seq Tools is installed, you can test the functionality of all methods by running

sortseq_tolls run_tests

Documentation

The commands used to perform the analysis in Ireland & Kinney (2016) are described here [link to analysis.rst]. Documentation on each of the Sort-Seq Tools functions is provided at `sortseq_tools documentation`_

.. _`sortseq_tools documentation`: http://jbkinney.github.io/sortseq



