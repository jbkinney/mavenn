#!/usr/bin/env python

'''A script which calculates the mutual information between base identity
    and batch.'''

from __future__ import division
#Our standard Modules
import argparse
import numpy as np
import sys
import csv
#Our miscellaneous functions
#This module will allow us to easily tally the letter counts at a particular position
import pandas as pd
import utils as utils
import qc as qc
import io_local as io
import profile_ct as profile_ct
import info as info
import pdb
from utils import check, ControlledError, handle_errors
from mpathic import SortSeqError

class ProfileInfo:

    """
    Profile Info computes the mutual information (in bits), at each position, between the character and the bin number.

    Parameters
    ----------

    dataset_df: (pandas dataframe)
        Input data frame

    err: (boolean)
        If true, include error estimates in computed mutual information

    method: (string)
        method used in computation. Valid choices inlcude: 'naive','tpm','nsb'.

    pseudocount: (float)
        pseudocount used to compute information values

    start: (int)
        An integer specifying the sequence start position

    end: (int)
        An integer specifying the sequence end position

    Returns
    -------
    info_df: (pandas dataframe)
        dataframe containing results.

    """

    @handle_errors
    def __init__(self, dataset_df = None, err=False, method='naive',
                 pseudocount=1.0, start=0, end=None):

        # set attributes to parameters

        self.dataset_df = dataset_df
        self.err = err
        self.method = method
        self.pseudocount = pseudocount
        self.start = start
        self.end = end

        # df containing result, declared now, set below.
        self.info_df = None

        # validate/check inputs
        self._input_check()

        #filename = './mpathic/data/formatted/dms/dms_1_formatted.txt'
        #dataset_df = pd.read_csv(filename, delim_whitespace=True, dtype={'seqs': str, 'batch': int})

        # Get number of bins
        bin_cols = [c for c in dataset_df.columns if qc.is_col_type(c,'ct_')]
        #print(bin_cols)
        if not len(bin_cols) >= 2:
            raise SortSeqError('Information profile requires at least 2 bins.')
        bins = [int(c.split('_')[1]) for c in bin_cols]
        num_bins = len(bins)

        # Get number of characters
        seq_cols = [c for c in dataset_df.columns if qc.is_col_type(c,'seqs')]
        if not len(seq_cols)==1:
            raise SortSeqError('Must be only one seq column.')
        seq_col = seq_cols[0]
        seqtype = qc.colname_to_seqtype_dict[seq_col]
        alphabet = qc.seqtype_to_alphabet_dict[seqtype]
        ct_cols = ['ct_'+a for a in alphabet]
        num_chars = len(alphabet)

        # Get sequence length and check start, end numbers
        num_pos = len(dataset_df[seq_col][0])
        if not (0 <= start < num_pos):
            raise SortSeqError('Invalid start==%d, num_pos==%d'%(start,num_pos))
        if end is None:
            end = num_pos
        elif (end > num_pos):
            raise SortSeqError('Invalid end==%d, num_pos==%d'%(end,num_pos))
        elif end <= start:
            raise SortSeqError('Invalid: start==%d >= end==%d'%(start,end))

        # Record positions in new dataframe
        counts_df = profile_ct.main(dataset_df)
        info_df = counts_df.loc[start:(end-1),['pos']].copy() # rows from start:end
        info_df['info'] = 0.0
        if err:
            info_df['info_err'] = 0.0

        # Fill in 3D array of counts
        ct_3d_array = np.zeros([end-start, num_chars, num_bins])
        for i, bin_num in enumerate(bins):

            # Compute counts
            counts_df = profile_ct.main(dataset_df, bin=bin_num)

            # Fill in counts table
            ct_3d_array[:,:,i] = counts_df.loc[start:(end-1),ct_cols].astype(float)

        # Compute mutual information for each position
        for i in range(end-start): # i only from start:end

            # Get 2D counts
            nxy = ct_3d_array[i,:,:]
            assert len(nxy.shape) == 2

            # Compute mutual informaiton
            if err:
                mi, mi_err = info.estimate_mutualinfo(nxy,err=True,\
                    method=method,pseudocount=pseudocount)
                info_df.loc[i+start,'info'] = mi
                info_df.loc[i+start,'info_err'] = mi_err
            else:
                mi = info.estimate_mutualinfo(nxy,err=False,\
                    method=method,pseudocount=pseudocount)
                info_df.loc[i+start,'info'] = mi

        # Validate info dataframe
        info_df = qc.validate_profile_info(info_df,fix=True)
        #return info_df
        self.info_df = info_df

    def _input_check(self):

        # check that dataset_df is valid
        if self.dataset_df is None:
            raise ControlledError(" Profile info requires pandas dataframe as input dataframe. Entered df was 'None'.")

        elif self.dataset_df is not None:
            check(isinstance(self.dataset_df, pd.DataFrame),
                  'type(df) = %s; must be a pandas dataframe ' % type(self.dataset_df))

            # validate dataset
            check(pd.DataFrame.equals(self.dataset_df, qc.validate_dataset(self.dataset_df)),
                  " Input dataframe failed quality control, \
                  please ensure input dataset has the correct format of an mpathic dataframe ")

        # check that attribute err is of type boolean
        check(isinstance(self.err,bool), 'type(err) = %s; must be a boolean ' % type(self.err))

        # method method is string ...
        check(isinstance(self.method,str), 'type(method) = %s; must be a string ' % type(self.method))

        # ... and string value is valid
        valid_method_choices = ['naive', 'tpm', 'nsb']
        check(self.method in valid_method_choices, 'method = %s; must be in %s' % (self.method, valid_method_choices))

        # check pseudocount is float
        check(isinstance(self.pseudocount,float),'type(pseudocount) = %s; must be a float ' % type(self.pseudocount))

        check(isinstance(self.start, int),
              'type(start) = %s; must be of type int ' % type(self.start))

        check(self.start >= 0, "start = %d must be a positive integer " % self.start)

        if self.end is not None:
            check(isinstance(self.end, int),
                  'type(end) = %s; must be of type int ' % type(self.end))





