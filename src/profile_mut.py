#!/usr/bin/env python
'''
This class calculates the fractional number of character occurances at each position within the set of sequences
passed. Specificially, it computes the mutation rate (0.0 to 1.0) at each position. Mutation rate is defined as
1.0 minus the maximum character frequency at a position. Errors are estimated using bionomial uncertainty

'''
from __future__ import division
import numpy as np
import sys
import pandas as pd
import qc as qc
import io_local as io
import profile_ct as profile_ct
import pdb
from mpathic import SortSeqError
from utils import ControlledError, handle_errors, check



class ProfileMut:
    """

    Parameters
    ----------

    dataset_df: (pandas dataframe)
        Input data frame containing a valid dataset.

    bin: (int)
        A bin number specifying which counts to use

    start: (int)
        An integer specifying the sequence start position

    end: (int)
        An integer specifying the sequence end position

    err: (boolean)
        If true, include error estimates in computed mutual information

    Returns
    -------
    mut_df: (pandas data frame)
            A pandas dataframe containing results.
    """

    @handle_errors
    def __init__(self, dataset_df=None, bin=None, start=0, end=None, err=False):


        # set attributes

        self.dataset_df = dataset_df
        self.bin = bin
        self.start = start
        self.end = end
        self.err = err
        self.mut_df = None

        # do input checks
        self._input_checks()

        #filename = './mpathic/examples/data_set_simulated.txt'
        #dataset_df = pd.read_csv(filename, delim_whitespace=True, dtype={'seqs': str, 'batch': int})
    
        # Compute counts
        counts_df = profile_ct.main(dataset_df, bin=bin, start=start, end=end)
    
        # Create columns for profile_freqs table
        ct_cols = [c for c in counts_df.columns if qc.is_col_type(c, 'ct_')]
    
        # Record positions in new dataframe
        mut_df = counts_df[['pos']].copy()
    
        # Compute mutation rate across counts
        max_ct = counts_df[ct_cols].max(axis=1)
        sum_ct = counts_df[ct_cols].sum(axis=1)
        mut = 1.0 - (max_ct / sum_ct)
        mut_df['mut'] = mut
    
        # Computation of error rate is optional
        if err:
            mut_err = np.sqrt(mut * (1.0 - mut) / sum_ct)
            mut_df['mut_err'] = mut_err
    
        # Figure out which alphabet the cts dataframe specifies
        alphabet = ''.join([c.split('_')[1] for c in ct_cols])
        seqtype = qc.alphabet_to_seqtype_dict[alphabet]
        wt_col = qc.seqtype_to_wtcolname_dict[seqtype]
    
        # Compute WT base at each position
        mut_df[wt_col] = 'X'
        for col in ct_cols:
            indices = (counts_df[col] == max_ct).values
            mut_df.loc[indices, wt_col] = col.split('_')[1]
    
        # Validate as counts dataframe
        mut_df = qc.validate_profile_mut(mut_df, fix=True)
        self.mut_df = mut_df

    def _input_checks(self):

        # check that dataset_df is valid
        if self.dataset_df is None:
            raise ControlledError(
                " Profile info requires pandas dataframe as input dataframe. Entered df was 'None'.")

        elif self.dataset_df is not None:
            check(isinstance(self.dataset_df, pd.DataFrame),
                  'type(df) = %s; must be a pandas dataframe ' % type(self.dataset_df))

            # validate dataset
            check(pd.DataFrame.equals(self.dataset_df, qc.validate_dataset(self.dataset_df)),
                  " Input dataframe failed quality control, \
                  please ensure input dataset has the correct format of an mpathic dataframe ")

        if self.bin is not None:
            # check bin is int
            check(isinstance(self.bin, int),
                  'type(bin) = %s; must be a int ' % type(self.bin))

        check(isinstance(self.start, int),
              'type(start) = %s; must be of type int ' % type(self.start))

        check(self.start >= 0, "start = %d must be a positive integer " % self.start)

        if self.end is not None:
            check(isinstance(self.end, int),
                  'type(end) = %s; must be of type int ' % type(self.end))

        # check that attribute err is of type boolean
        check(isinstance(self.err, bool), 'type(err) = %s; must be a boolean ' % type(self.err))
