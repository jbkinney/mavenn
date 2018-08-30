"""A script which returns the mutual information between the predictions of a
    model and a test data set."""

import numpy as np
# Our miscellaneous functions
# This module will allow us to easily tally the letter counts at a particular position

#import utils as utils
from mpathic.src import utils
#import EstimateMutualInfoforMImax as EstimateMutualInfoforMImax
from mpathic.src import EstimateMutualInfoforMImax
#import qc as qc
from mpathic.src import qc
from mpathic.src.utils import check,ControlledError,handle_errors
#import numerics as numerics
from mpathic.src import numerics
from mpathic import SortSeqError
import pandas as pd
from mpathic.src import io_local as io
import matplotlib.pyplot as plt

class PredictiveInfo:


    """

        Parameters
        ----------

        data_df: (pandas data frame)
         Dataframe containing several columns representing \n
         bins and sequence column. The integer values in bins \n
         represent the occurrence of the sequence that bin.

        model_df: (pandas dataframe)
            The dataframe containing a model of the binding \n
            energy and a wild type sequence

        start: (int)
            Starting position of the sequence.

        end: (int)
            end position of the sequence.

        err: (bool)
            boolean variable which indiciates the inclusion of
            error in the mutual information estimate if true

        coarse_graining_level: (int)
            Speed computation by coarse-graining model predictions

    """

    @handle_errors
    def __init__(self,
                 data_df,
                 model_df,
                 start=0,
                 end=None,
                 err=False,
                 coarse_graining_level=0,
                 rsquared=False,
                 return_freg=False):

        self.data_df = data_df
        self.model_df = model_df
        self.start = start
        self.end = end
        self.err = err
        self.coarse_graining_level = coarse_graining_level

        self.out_MI = None
        self.out_std = None


        self._input_checks()

        dicttype, modeltype = qc.get_model_type(self.model_df)
        seq_cols = qc.get_cols_from_df(self.data_df, 'seqs')
        if not len(seq_cols) == 1:
            raise SortSeqError('Dataframe has multiple seq cols: %s' % str(seq_cols))
        seq_dict, inv_dict = utils.choose_dict(dicttype, modeltype=modeltype)
        # set name of sequences column based on type of sequence
        type_name_dict = {'dna': 'seq', 'rna': 'seq_rna', 'protein': 'seq_pro'}
        seq_col_name = type_name_dict[dicttype]
        # Cut the sequences based on start and end, and then check if it makes sense
        if (self.start != 0 or self.end):
            self.data_df.loc[:, seq_col_name] = self.data_df.loc[:, seq_col_name].str.slice(self.start, self.end)
            if modeltype == 'MAT':
                if len(self.data_df.loc[0, seq_col_name]) != len(self.model_df.loc[:, 'pos']):
                    print('predictive info class: BP lengths: ',len(self.data_df.loc[0, seq_col_name])," ",len(self.model_df.loc[:, 'pos']))
                    raise SortSeqError('model length does not match dataset length')
            elif modeltype == 'NBR':
                if len(self.data_df.loc[0, seq_col_name]) != len(self.model_df.loc[:, 'pos']) + 1:
                    raise SortSeqError('model length does not match dataset length')
        col_headers = utils.get_column_headers(self.data_df)
        if 'ct' not in self.data_df.columns:
            self.data_df['ct'] = data_df[col_headers].sum(axis=1)
            self.data_df = self.data_df[self.data_df.ct != 0]
        if not self.end:
            seqL = len(self.data_df[seq_col_name][0]) - self.start
        else:
            seqL = self.end - self.start
            self.data_df = self.data_df[self.data_df[seq_col_name].apply(len) == (seqL)]
        # make a numpy array out of the model data frame
        model_df_headers = ['val_' + str(inv_dict[i]) for i in range(len(seq_dict))]
        value = np.transpose(np.array(self.model_df[model_df_headers]))
        # now we evaluate the expression of each sequence according to the model.
        seq_mat, wtrow = numerics.dataset2mutarray(self.data_df.copy(), modeltype)
        temp_df = self.data_df.copy()
        # AT: what is this line trying to do?
        temp_df['val'] = numerics.eval_modelmatrix_on_mutarray(np.array(self.model_df[model_df_headers]), seq_mat, wtrow)
        temp_sorted = temp_df.sort_values(by='val')
        temp_sorted.reset_index(inplace=True, drop=True)
        # we must divide by the total number of counts in each bin for the MI calculator
        # temp_sorted[col_headers] = temp_sorted[col_headers].div(temp_sorted['ct'],axis=0)
        if return_freg:
            fig, ax = plt.subplots()
            MI, freg = EstimateMutualInfoforMImax.alt4(temp_sorted, coarse_graining_level=coarse_graining_level,
                                                       return_freg=return_freg)
            plt.imshow(freg, interpolation='nearest', aspect='auto')

            plt.savefig(return_freg)
        else:
            MI = EstimateMutualInfoforMImax.alt4(temp_sorted, coarse_graining_level=coarse_graining_level,
                                                 return_freg=return_freg)
        if not self.err:
            Std = np.NaN
        else:
            data_df_for_sub = self.data_df.copy()
            sub_MI = np.zeros(15)
            for i in range(15):
                sub_df = data_df_for_sub.sample(int(len(data_df_for_sub.index) / 2))
                sub_df.reset_index(inplace=True, drop=True)
                sub_MI[i], sub_std = PredictiveInfo(sub_df, model_df, err=False)
            Std = np.std(sub_MI) / np.sqrt(2)
        if rsquared:
            #return (1 - 2 ** (-2 * MI)), (1 - 2 ** (-2 * Std))
            self.out_MI,self.out_std = (1 - 2 ** (-2 * MI)), (1 - 2 ** (-2 * Std))
        else:
            #return MI, Std
            self.out_MI, self.out_std = MI, Std


    def _input_checks(self):

        # data_df validation
        if self.data_df is None:
            raise ControlledError(
                " The Predictive Info class requires pandas dataframe as input dataframe. Entered data_df was 'None'.")

        elif self.data_df is not None:
            check(isinstance(self.data_df, pd.DataFrame),
                  'type(data_df) = %s; must be a pandas dataframe ' % type(self.data_df))

        # validate data_df
        check(pd.DataFrame.equals(self.data_df, qc.validate_dataset(self.data_df)), " Input dataframe fails quality control, please ensure input dataframe has the correct format of an mpathic dataframe ")


        # model validation
        if self.model_df is None:
            raise ControlledError(
                " The Predictive info class requires pandas dataframe as input model dataframe. Entered model_df was 'None'.")

        elif self.model_df is not None:
            check(isinstance(self.model_df, pd.DataFrame),
                  'type(model_df) = %s; must be a pandas dataframe ' % type(self.model_df))


        # validate model df
        check(pd.DataFrame.equals(self.model_df, qc.validate_model(self.model_df)),
              " Model dataframe failed quality control, \
                                please ensure input model dataframe has the correct format of an mpathic dataframe ")


        # check that start is an integer
        check(isinstance(self.start, int),
              'type(start) = %s; must be of type int ' % type(self.start))

        check(self.start >= 0, "start = %d must be a positive integer " % self.start)

        if self.end is not None:
            check(isinstance(self.end, int),
                  'type(end) = %s; must be of type int ' % type(self.end))

        # check that verbose is a boolean
        check(isinstance(self.err, bool),
              'type(err) = %s; must be of type bool ' % type(self.err))

        check(isinstance(self.coarse_graining_level, int),
                  'type(coarse_graining_level) = %s; must be of type int ' % type(self.coarse_graining_level))

