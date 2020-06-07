from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split as sk_train_test_split
from mavenn.src.error_handling import handle_errors


@handle_errors
def train_test_split(sequences, values, test_size=0.2, random_state=0):
    """
    Uses sklearns train_test_split method to split data in to training and
    testing sets

    parameters
    ----------

    sequences: (array-like)
        List of biological sequences

    values: (array-like)
        Biological function values of sequences

    test_size: (float in [0,1])
        Specifies fraction of test set.

    random_state: (integer)
        Specifies seed to for sklearns train_test_split method

    returns
    -------
    x_train, x_test, y_train, y_test: (array-like)
        returns arrays of sequences and values split into training
        and test sets

    """

    # TODO: is there a need to do input checks here?

    x_train, x_test, y_train, y_test = sk_train_test_split(sequences,
                                                           values,
                                                           test_size=test_size,
                                                           random_state=random_state)
    return x_train, x_test, y_train, y_test


# Fit a label encoder and a onehot encoder
bases = ["A","C","G","U"]
label_encoder = LabelEncoder()
label_encoder.fit(bases)
tmp = label_encoder.transform(bases)
tmp = tmp.reshape(len(tmp), 1)
onehot_encoder = OneHotEncoder(sparse = False)
onehot_encoder.fit(tmp)


# Encode sequence into onehot
def onehot_sequence(sequence, lab_encoder = label_encoder, one_encoder = onehot_encoder):
    """Sequence as a string"""
    tmp = lab_encoder.transform(list(sequence))
    tmp = tmp.reshape(len(tmp),1)
    tmp = one_encoder.transform(tmp)
    return tmp
