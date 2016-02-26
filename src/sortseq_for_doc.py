#!/usr/bin/env python2.7

''' Primary function for sortseq_tools.ools. Currently supports: 

simulate_library
simulate_sublib
simulate_sortseq_tools
simulate_selection
simulate_mpra
'''

from __future__ import division
import numpy as np
import scipy as sp
import argparse
import sys
import csv

# Create argparse parser. 
parser = argparse.ArgumentParser()

# All functions can specify and output file. Default is stdout.
parser.add_argument('-o','--out',default=False,help='Output location/type, by default it writes to standard output, if a file name is supplied it will write to a text file')

# Add various subcommands individually viva subparsers
subparsers = parser.add_subparsers()

# preprocess
import sortseq_tools.preprocess as preprocess
preprocess.add_subparser(subparsers)

#profile_mutrate
import sortseq_tools.profile_mut as profile_mut
profile_mut.add_subparser(subparsers)

#learn_matrix
import sortseq_tools.learn_matrix as learn_matrix
learn_matrix.add_subparser(subparsers)

#predictiveinfo
import sortseq_tools.predictiveinfo as predictiveinfo
predictiveinfo.add_subparser(subparsers)

#profile_info
import sortseq_tools.profile_info as profile_info
profile_info.add_subparser(subparsers)

#Scan
import sortseq_tools.scan_model as scan_model
scan_model.add_subparser(subparsers)

#simualte_library
import sortseq_tools.simulate_library as simulate_library
simulate_library.add_subparser(subparsers)

#simulate_sort
import sortseq_tools.simulate_sort as simulate_sort
simulate_sort.add_subparser(subparsers)

# #simulate_evaluate
# import sortseq_tools.simulate_evaluate as simulate_evaluate
# simulate_evaluate.add_subparser(subparsers)

#simulate_sort
import sortseq_tools.simulate_expression as simulate_expression
simulate_expression.add_subparser(subparsers)

# Final incantiation needed for this to work









