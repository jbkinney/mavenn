from __future__ import division
import pandas as pd
from mavenn.src.error_handling import check, handle_errors


@handle_errors
def validate_input(df):
    """
    Checks to make sure that the input dataframe, df, contains
    sequences and values that are valid for mavenn. Sequences
    must all be of the same length and values have to be
    integers or floats and not nan or string etc.

    parameters
    ----------

    df: (dataframe)
        A pandas dataframe containing two columns: (i) sequences and (ii)
        values.

    returns
    -------
    out_df: (dataframe)
        A cleaned-up version of df (if possible).
    """

    # check that df is a valid dataframe
    check(isinstance(df, pd.DataFrame),
          'Input data needs to be a valid pandas dataframe, ' 
          'input entered: %s' % type(df))

    # create copy of df so we don't overwrite the user's data
    out_df = df.copy()

    # make sure the input df has only sequences and values
    # and no additional columns
    check(out_df.shape[1] == 2, 'Input dataframe must only have 2 columns, '
                                'sequences and values. Entered # columns %d' % len(out_df.columns))

    # check that 'sequences' and 'values columns are part of the df columns
    check('sequence' in out_df.columns, 'Column containing sequences must be named "sequence" ')
    check('values' in out_df.columns, 'Column containing values must be named "values" ')

    # TODO: check that sequence column is of type string and values column is float or int
    # TODO: need to check that sequences are of the same length, but this could take a lot of time checking

    # return cleaned-up out_df
    out_df = out_df.dropna().copy()
    return out_df


