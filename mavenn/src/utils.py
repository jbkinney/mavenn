from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from mavenn.src.error_handling import handle_errors

# global variables needed for sklearns one-hot encoder
# Fit a label encoder and a onehot encoder
bases = ["A","C","G","U"]
label_encoder = LabelEncoder()
label_encoder.fit(bases)
tmp = label_encoder.transform(bases)
tmp = tmp.reshape(len(tmp), 1)
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder.fit(tmp)


@handle_errors
def onehot_sequence(sequence, lab_encoder = label_encoder, one_encoder = onehot_encoder):
    """
    Encodes a single sequence into onehot vector
    """
    tmp = lab_encoder.transform(list(sequence))
    tmp = tmp.reshape(len(tmp),1)
    tmp = one_encoder.transform(tmp)
    return tmp


@handle_errors
def onehot_encode_array(data, bases_dict):

    """
    one-hot encode sequences in batches in a vectorized way

    """

    sequence_length = len(data[0])

    ohe_single_batch_size = 10000
    # container list for batches of oh-encoded sequences
    input_seqs_ohe_batches = []

    # partitions of batches
    ohe_batches = np.arange(0, len(data), ohe_single_batch_size)
    for ohe_batch_index in range(len(ohe_batches)):
        if ohe_batch_index == len(ohe_batches) - 1:
            # OHE remaining sequences (that are smaller than batch size)
            input_seqs_ohe_batches.append(
                onehot_sequence(''.join(data[ohe_batches[ohe_batch_index]:]))
                    .reshape(-1, sequence_length, len(bases_dict)))
        else:
            # OHE sequences in batches
            input_seqs_ohe_batches.append(onehot_sequence(
                ''.join(data[ohe_batches[ohe_batch_index]:ohe_batches[ohe_batch_index + 1]]))
                                          .reshape(-1, sequence_length, len(bases_dict)))

    # this array will contain the one-hot encoded sequences
    input_seqs_ohe = np.array([])

    # concatenate all the oh-encoded batches
    for batch_index in range(len(input_seqs_ohe_batches)):
        input_seqs_ohe = np.concatenate([input_seqs_ohe, input_seqs_ohe_batches[batch_index]
                                             .ravel()]).copy()

    # reshape so that shape of oh-encoded array is [number samples, sequence_length*alphabet_dict]
    input_seqs_ohe = input_seqs_ohe.reshape(len(data), sequence_length * len(bases_dict)).copy()

    return input_seqs_ohe
