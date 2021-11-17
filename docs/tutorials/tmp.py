## PACKAGE THIS FUNCTION INTO MAVENN
def split_dataset(data_df,
                  set_col='set',
                  train_set_name='training',
                  val_set_name='validation',
                  test_set_name='test'):
    """
    Splits dataset into
        (1) `trainval_df`: training + validation set
        (2) `train_df`: test set
    based on the value of the column `set_col`, which is then dropped. Also
    adds a `val_set_name` column to trainval_df indicating which data is to be
    reserved for validation (as opposed to gradient descent).

    Parameters
    ----------
    data_df: (pd.DataFrame)
        Dataset to split

    set_col: (str)
        Column of data_df indicating training, validation, or test set

    train_set_name: (str)
        Value in data_df[set_col] indicating allocation to training set

    val_set_name: (str)
        Value in data_df[set_col] indicating allocation to validation set

    test_set_name: (str)
        Value in data_df[set_col] indicating allocation to test set

    Returns
    -------
    trainval_df: (pd.DataFrame)
        Training + validation dataset. Contains a column named `val_set_name`
        indicating whether a row is allocated to the training or validation
        set.

    test_df: (pd.DataFrame)
        Test dataset.
    """

    # Specify training + validation sets
    trainval_ix = data_df[set_col].isin([train_set_name, val_set_name])
    trainval_df = data_df[trainval_ix].copy().reset_index(drop=True)
    trainval_df.insert(loc=0,
                       column=val_set_name,
                       value=trainval_df[set_col].eq(val_set_name))
    trainval_df.drop(columns=set_col, inplace=True)

    # Specify test set
    test_ix = data_df[set_col].eq(test_set_name)
    test_df = data_df[test_ix].copy().reset_index(drop=True)
    test_df.drop(columns=set_col, inplace=True)

    # return
    return trainval_df, test_df
