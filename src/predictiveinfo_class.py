"""A script which returns the mutual information between the predictions of a
    model and a test data set."""

import numpy as np
# Our miscellaneous functions
# This module will allow us to easily tally the letter counts at a particular position

import utils as utils
import EstimateMutualInfoforMImax as EstimateMutualInfoforMImax
import qc as qc
import numerics as numerics
from mpathic import SortSeqError
import io_local as io
import matplotlib.pyplot as plt

class predictiveinfo_class:

    def __init__(self,data_df, model_df,start=0, end=None, err=False, coarse_graining_level=0,rsquared=False, return_freg=False):

        dicttype, modeltype = qc.get_model_type(model_df)
        seq_cols = qc.get_cols_from_df(data_df, 'seqs')
        if not len(seq_cols) == 1:
            raise SortSeqError('Dataframe has multiple seq cols: %s' % str(seq_cols))
        seq_dict, inv_dict = utils.choose_dict(dicttype, modeltype=modeltype)
        # set name of sequences column based on type of sequence
        type_name_dict = {'dna': 'seq', 'rna': 'seq_rna', 'protein': 'seq_pro'}
        seq_col_name = type_name_dict[dicttype]
        # Cut the sequences based on start and end, and then check if it makes sense
        if (start != 0 or end):
            data_df.loc[:, seq_col_name] = data_df.loc[:, seq_col_name].str.slice(start, end)
            if modeltype == 'MAT':
                if len(data_df.loc[0, seq_col_name]) != len(model_df.loc[:, 'pos']):
                    print('predictive info class: BP lengths: ',len(data_df.loc[0, seq_col_name])," ",len(model_df.loc[:, 'pos']))
                    raise SortSeqError('model length does not match dataset length')
            elif modeltype == 'NBR':
                if len(data_df.loc[0, seq_col_name]) != len(model_df.loc[:, 'pos']) + 1:
                    raise SortSeqError('model length does not match dataset length')
        col_headers = utils.get_column_headers(data_df)
        if 'ct' not in data_df.columns:
            data_df['ct'] = data_df[col_headers].sum(axis=1)
        data_df = data_df[data_df.ct != 0]
        if not end:
            seqL = len(data_df[seq_col_name][0]) - start
        else:
            seqL = end - start
        data_df = data_df[data_df[seq_col_name].apply(len) == (seqL)]
        # make a numpy array out of the model data frame
        model_df_headers = ['val_' + str(inv_dict[i]) for i in range(len(seq_dict))]
        value = np.transpose(np.array(model_df[model_df_headers]))
        # now we evaluate the expression of each sequence according to the model.
        seq_mat, wtrow = numerics.dataset2mutarray(data_df.copy(), modeltype)
        temp_df = data_df.copy()
        # AT: what is this line trying to do?
        temp_df['val'] = numerics.eval_modelmatrix_on_mutarray(np.array(model_df[model_df_headers]), seq_mat, wtrow)
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
        if not err:
            Std = np.NaN
        else:
            data_df_for_sub = data_df.copy()
            sub_MI = np.zeros(15)
            for i in range(15):
                sub_df = data_df_for_sub.sample(int(len(data_df_for_sub.index) / 2))
                sub_df.reset_index(inplace=True, drop=True)
                sub_MI[i], sub_std = self.__init__(
                    sub_df, model_df, err=False)
            Std = np.std(sub_MI) / np.sqrt(2)
        if rsquared:
            return (1 - 2 ** (-2 * MI)), (1 - 2 ** (-2 * Std))
        else:
            return MI, Std
