'''A script which adds a predicted energy column to an input table. This is
    generated based on a energy model the user provides.'''
from __future__ import division
import Models as Models
import utils as utils
import qc as qc
import io_local as io
from mpathic import SortSeqError
from mpathic import shutthefuckup
import fast

class EvaluateModel:

    """

    Parameters
    ----------

    dataset_df: (pandas dataframe)
        Input dataset data frame
    model_df: (pandas dataframe)
        Model dataframe
    left: (int)
        Seq position at which to align the left-side of the model. \n
        Defaults to position determined by model dataframe.

    right: (int)
        Seq position at which to align the right-side of the model. \n
        Defaults to position determined by model dataframe.

    """

    def __init__(self,dataset_df, model_df, left=None, right=None):



        self.dataset_with_values = None
        self.out_df = None

        qc.validate_dataset(dataset_df)
        qc.validate_model(model_df)

        seqtype, modeltype = qc.get_model_type(model_df)
        seqcol = qc.seqtype_to_seqcolname_dict[seqtype]

        if not ((left is None) or (right is None)):
            raise SortSeqError('Cannot set both left and right at same time.')
        if not (left is None):
            start = left
            end = start + model_df.shape[0] + (1 if modeltype == 'NBR' else 0)
        elif not (right is None):
            end = right
            start = end - model_df.shape[0] - (1 if modeltype == 'NBR' else 0)
        else:
            start = model_df['pos'].values[0]
            end = model_df['pos'].values[-1] + (2 if modeltype == 'NBR' else 1)
        assert start < end

        # Validate start and end positions
        seq_length = len(dataset_df[seqcol][0])
        if start < 0:
            raise SortSeqError('Invalid start=%d' % start)
        if end > seq_length:
            raise SortSeqError('Invalid end=%d for seq_length=%d' % (end, seq_length))

        # select target sequence region
        out_df = dataset_df.copy()
        out_df.loc[:, seqcol] = out_df.loc[:, seqcol].str.slice(start, end)

        # Create model object of correct type
        if modeltype == 'MAT':
            mymodel = Models.LinearModel(model_df)
        elif modeltype == 'NBR':
            mymodel = Models.NeighborModel(model_df)
        else:
            raise SortSeqError('Unrecognized model type %s' % modeltype)

        # Compute values
        out_df['val'] = mymodel.evaluate(out_df)
        # Validate dataframe and return
        #print(out_df['val'])
        #return qc.validate_dataset(out_df, fix=True)
        self.dataset_with_values = qc.validate_dataset(out_df, fix=True)
        self.out_df = out_df

