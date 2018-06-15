#!/usr/bin/env python

'''Simulate cell sorting based on expression'''
from __future__ import division
import argparse
import numpy as np
import scipy as sp
import pandas as pd
import sys
import Models as Models
import utils as utils
import io_local as io
import qc as qc
from evaluate_model import EvaluateModel
from mpathic import SortSeqError
from utils import check, handle_errors, ControlledError


class SimulateSort:

    """

    Parameters
    ----------

    df: (pandas dataframe)
        Input data frame.

    mp: (pandas dataframe)
        Model data frame.

    noisetype: (string, None)
        Noise parameter string indicating what type of \n
        noise to include. Valid choices include None, 'Normal', 'LogNormal', 'Plasmid'

    npar: (list)
        parameters to go with noisetype. E.g. for \n
        noisetype 'Normal', npar must contain the width of the normal distribution

    nbins: (int)
        Number of bins that the different variants will get sorted into.

    sequence_library: (bool)
        A value of True corresponds to simulating sequencing the library in bin zero

    start: (int)
        Position to start analyzed region

    end: (int)
        Position to end analyzed region

    chunksize: (int)
        This represents the size of chunk the data frame df will be traversed over.


    Attributes
    ----------

    output_df: (pandas data frame)
        contains the output of the simulate_sort constructor
    """

    # def __init__(self,df,mp,noisetype,npar,nbins,sequence_library=True,start=0,end=None,chunksize=50000):
    @handle_errors
    def __init__(self, df=None, mp=None, noisetype='None', npar=[0.2], nbins=3, sequence_library=True, start=0,
                 end=None, chunksize=10):

        # set attributes of class
        self.df = df
        self.mp = mp
        self.noisetype = noisetype
        self.npar = npar
        self.nbins = nbins
        self.sequence_library = sequence_library
        self.start = start
        self.end = end
        self.chunksize = chunksize

        # validate inputs
        self._input_check()

        # determine cutoffs for bins now
        # do progressive sum to try to find cutoffs so their will be equal numbers in each bin

        # Determine model type to use for noise
        if noisetype == 'LogNormal':
            NoiseModelSort = Models.LogNormalNoise(npar)
        elif noisetype == 'Normal':
            NoiseModelSort = Models.NormalNoise(npar)
        elif noisetype == 'None':
            NoiseModelSort = Models.NormalNoise([1e-16])
        elif noisetype == 'Plasmid':
            NoiseModelSort = Models.PlasmidNoise()
        else:
            NoiseModelSort = Models.CustomModel(noisetype, npar)
        i = 0
        output_df = pd.DataFrame()

        # split the dataframe according to chunksize
        df = np.array_split(df, chunksize)
        for chunk in df:

            chunk.reset_index(inplace=True, drop=True)
            chunk = EvaluateModel(chunk, mp, left=start, right=None).out_df
    
            # Apply noise to our calculated energies
            noisyexp, listnoisyexp = NoiseModelSort.genlist(chunk)
            if i == 0:
                # Determine Expression Cutoffs for bins
                noisyexp.sort()
                val_cutoffs = list(
                    noisyexp[np.linspace(0, len(noisyexp), nbins, endpoint=False, dtype=int)])
                val_cutoffs.append(np.inf)
                val_cutoffs[0] = -np.inf
            #print val_cutoffs
            # Determine Expression Cutoffs for bins
            seqs_arr = np.zeros([len(listnoisyexp), nbins], dtype=int)
            # split sequence into bins based on calculated cutoffs
            for i, entry in enumerate(listnoisyexp):
                seqs_arr[i, :] = np.histogram(entry, bins=val_cutoffs)[0]
            col_labels = ['ct_' + str(i + 1) for i in range(nbins)]
            if sequence_library:
                chunk.loc[:, 'ct_0'] = utils.sample(chunk.loc[:, 'ct'], int(chunk.loc[:, 'ct'].sum() / nbins))
            temp_output_df = pd.concat([chunk, pd.DataFrame(seqs_arr, columns=col_labels)], axis=1)
            col_labels = utils.get_column_headers(temp_output_df)
            # temp_output_df['ct'] = temp_output_df[col_labels].sum(axis=1)
            temp_output_df.drop('val', axis=1, inplace=True)
            #print output_df.shape
            output_df = pd.concat([output_df, temp_output_df], axis=0).copy()
            i = i + 1

        output_df['ct'] = output_df[col_labels].sum(axis=1)
        output_df.reset_index(inplace=True, drop=True)

        self.output_df = output_df

    def _input_check(self):
        """
        private method that validates all parameters
        """

        # check that input df is of type pandas dataframe
        if self.df is None:
            raise ControlledError(" Simulate Sort Requires pandas dataframe as input dataframe. Entered df was 'None'.")
        elif self.df is not None:
            check(isinstance(self.df, pd.DataFrame),
                  'type(df) = %s; must be a pandas dataframe ' % type(self.df))

            # validate dataset
            check(pd.DataFrame.equals(self.df, qc.validate_dataset(self.df)),
                  " Input dataframe failed quality control, \
                  please ensure input dataset has the correct format of an mpathic dataframe ")

        # check model dataframe
        if self.mp is None:
            raise ControlledError(
                " Simulate Sort Requires pandas dataframe as model input. Entered model df was 'None'.")
        elif self.mp is not None:
            check(isinstance(self.mp, pd.DataFrame), 'type(mp) = %s; must be a pandas dataframe ' % type(self.mp))

            # validate dataset
            check(pd.DataFrame.equals(self.mp, qc.validate_model(self.mp)),
                  " Model dataframe failed quality control, \
                  please ensure model has the correct format of an mpathic model dataframe ")

        # check noisetype is string
        check(isinstance(self.noisetype,str),'type(noisetype) = %s; must be a string ' % type(self.noisetype))

        # check noisetype is valid
        valid_noisetype_values = ['LogNormal','Normal','None','Plasmid']
        check(self.noisetype in valid_noisetype_values,
              'noisetype = %s; must be in %s' % (self.noisetype, valid_noisetype_values))

        # ensure that npar is type list
        check(isinstance(self.npar,list),'type(npar) = %s; must be a list ' % type(self.npar))

        # for valid choice of noisetype, pick appropriate noise parameters
        if self.noisetype == 'Normal':
            if len(self.npar) != 1:
                raise SortSeqError(
                    'For a normal noise model, there must be one input parameter (width of normal distribution)')

        if self.noisetype == 'LogNormal':
            if len(self.npar) != 2:
                raise SortSeqError('''For a LogNormal noise model there must 
                         be 2 input parameters''')

        # ensure nbins is valid
        check(isinstance(self.nbins,int),'type(nbins) = %s; must be of type int ' % type(self.nbins))
        check(self.nbins > 1,'number of bins must be greater than 1, entered bins = %d' % self.nbins)

        # sequence library should be boolean
        check(isinstance(self.sequence_library, bool),
              'type(sequence_library) = %s; must be of type bool ' % type(self.sequence_library))

        # make sure start is of type int
        check(isinstance(self.start, int),
                  'type(start) = %s; must be of type int ' % type(self.start))

        # make sure end is of type int
        if self.end is not None:
            check(isinstance(self.end, int),
                  'type(end) = %s; must be of type int ' % type(self.end))

        # make sure end is of type int
        if self.chunksize is not None:
            check(isinstance(self.chunksize, int), 'type(chunksize) = %s; must be of type int ' % type(self.chunksize))



