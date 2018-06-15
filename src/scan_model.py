#!/usr/bin/env python

'''A script which accepts an model of binding energy and a wild type sequence.
    The script scans the model across the sequence, and generates an energy
    prediction for each starting position. It then sorts by best binding and
    displays all posibilities.'''
from __future__ import division
# Our standard Modules
import argparse
import numpy as np
import scipy as sp
import sys
# Our miscellaneous functions
import pandas as pd
from Bio import SeqIO
import utils as utils
import Models as Models
import io_local as io
import qc as qc

import fast

import re
import pdb
from mpathic import SortSeqError

class ScanModel:

    def __init__(self, model_df, contig_list, numsites=10, verbose=False):

        self.sitelist_df = None
        # Determine type of string from model
        qc.validate_model(model_df)
        seqtype, modeltype = qc.get_model_type(model_df)
        seq_dict, inv_dict = utils.choose_dict(seqtype, modeltype=modeltype)

        # Check that all characters are from the correct alphabet
        alphabet = qc.seqtype_to_alphabet_dict[seqtype]
        search_string = r"[^%s]" % alphabet
        for contig_str, contig_name, pos_offset in contig_list:
            if re.search(search_string, contig_str):
                raise SortSeqError( \
                    'Invalid character for seqtype %s found in %s.' % \
                    (seqtype, contig_name))

        # Create model object to evaluate on seqs
        if modeltype == 'MAT':
            model_obj = Models.LinearModel(model_df)
        elif modeltype == 'NBR':
            model_obj = Models.NeighborModel(model_df)

        # Create list of dataframes, one for each contig
        seq_col = qc.seqtype_to_seqcolname_dict[seqtype]
        L = model_obj.length
        sitelist_df = pd.DataFrame( \
            columns=['val', seq_col, 'left', 'right', 'ori', 'contig'])
        for contig_str, contig_name, pos_offset in contig_list:
            if len(contig_str) < L:
                continue
            this_df = pd.DataFrame( \
                columns=['val', seq_col, 'left', 'right', 'ori', 'contig'])
            num_sites = len(contig_str) - L + 1
            poss = np.arange(num_sites).astype(int)
            this_df['left'] = poss + pos_offset
            this_df['right'] = poss + pos_offset + L - 1
            # this_df[seq_col] = [contig_str[i:(i+L)] for i in poss]
            this_df[seq_col] = fast.seq2sitelist(contig_str, L)  # Cython
            this_df['ori'] = '+'
            this_df['contig'] = contig_name
            this_df['val'] = model_obj.evaluate(this_df[seq_col])
            sitelist_df = pd.concat([sitelist_df, this_df], ignore_index=True)

            # If scanning DNA, scan reverse-complement as well
            if seqtype == 'dna':
                # this_df[seq_col] = [qc.rc(s) for s in this_df[seq_col]]
                this_df[seq_col] = fast.seq2sitelist(contig_str, L, rc=True)  # Cython
                this_df['ori'] = '-'
                this_df['val'] = model_obj.evaluate(this_df[seq_col])
                sitelist_df = pd.concat([sitelist_df, this_df], ignore_index=True)

            # Sort by value and reindex
            sitelist_df.sort_values(by='val', ascending=False, inplace=True)
            sitelist_df.reset_index(drop=True, inplace=True)

            # Crop list at numsites
            if sitelist_df.shape[0] > numsites:
                sitelist_df.drop(sitelist_df.index[numsites:], inplace=True)

            if verbose:
                print ('.',
                sys.stdout.flush())

        if verbose:
            print('')
            sys.stdout.flush()

        # If no sites were found, raise error
        if sitelist_df.shape[0] == 0:
            raise SortSeqError( \
                'No full-length sites found within provided contigs.')

        sitelist_df = qc.validate_sitelist(sitelist_df, fix=True)
        #return sitelist_df
        self.sitelist_df = sitelist_df


