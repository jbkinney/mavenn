import mavenn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def load_olson_data_GB1():

    """
    helper function to turn data provided by Olson et al.
    into sequence-values dataframe.


    return
    ------
    gb1_df: (pd dataframe)
        dataframe containing sequences (single and double mutants)
        and their corresponding log enrichment values

    """

    # GB1 WT sequences
    WT_seq = 'QYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'

    # WT sequence library and selection counts.
    WT_input_count = 1759616
    WT_selection_count = 3041819

    # load double mutant data
    oslon_mutant_positions_data = pd.read_csv(mavenn.__path__[0] +
                                              '/examples/datafiles/gb1/oslon_data_double_mutants_ambler.csv',
                                              na_values="nan")

    # lists that will contain sequences and their values
    sequences = []
    enrichment = []

    for loop_index in range(len(oslon_mutant_positions_data)):

        # skip row 259455 containing, it contains bad data
        if loop_index == 259455:
            continue

        # get positions of double mutants in sequence
        mut_1_index = int(oslon_mutant_positions_data['Mut1 Position'][loop_index]) - 2
        mut_2_index = int(oslon_mutant_positions_data['Mut2 Position'][loop_index]) - 2

        # get identity of mutations
        mut_1 = oslon_mutant_positions_data['Mut1 Mutation'][loop_index]
        mut_2 = oslon_mutant_positions_data['Mut2 Mutation'][loop_index]

        # form full mutant sequence.
        temp_dbl_mutant_seq = list(WT_seq)
        temp_dbl_mutant_seq[mut_1_index] = mut_1
        temp_dbl_mutant_seq[mut_2_index] = mut_2

        if loop_index % 100000 == 0:
            print('generating data: %d out of %d' % (loop_index,len(oslon_mutant_positions_data)))

        # calculate enrichment for double mutant sequence sequence
        input_count = oslon_mutant_positions_data['Input Count'][loop_index]
        selection_count = oslon_mutant_positions_data['Selection Count'][loop_index]
        # added pseudocount to ensure log doesn't throw up
        temp_fitness = ((selection_count + 1) / input_count) / (WT_selection_count / WT_input_count)

        # append sequence
        sequences.append(''.join(temp_dbl_mutant_seq))
        enrichment.append(temp_fitness)

    # load single mutants data
    oslon_single_mutant_positions_data = pd.read_csv(mavenn.__path__[0] +
                                                     '/examples/datafiles/gb1/oslon_data_single_mutants_ambler.csv',
                                                     na_values="nan")

    for loop_index in range(len(oslon_single_mutant_positions_data)):
        mut_index = int(oslon_single_mutant_positions_data['Position'][loop_index]) - 2

        mut = oslon_single_mutant_positions_data['Mutation'][loop_index]

        temp_seq = list(WT_seq)
        temp_seq[mut_index] = mut

        # calculate enrichment for sequence
        input_count = oslon_single_mutant_positions_data['Input Count'][loop_index]
        selection_count = oslon_single_mutant_positions_data['Selection Count'][loop_index]
        # added pseudo count to ensure log doesn't throw up
        temp_fitness = ((selection_count + 1) / input_count) / (WT_selection_count / WT_input_count)

        sequences.append(''.join(temp_seq))
        enrichment.append(temp_fitness)

    enrichment = np.array(enrichment).copy()

    gb1_df = pd.DataFrame({'sequence': sequences, 'values': np.log(enrichment)}, columns=['sequence', 'values'])
    return gb1_df


# load data
gb1_df = load_olson_data_GB1()

# split data into train and test
x_train, x_test, y_train, y_test = train_test_split(gb1_df['sequence'].values, gb1_df['values'].values)

# load mavenn's GE model
GE_model = mavenn.GlobalEpistasisModel(X=x_train, y=y_train, model_type='additive', alphabet_dict='protein')
model = GE_model.define_model()
GE_model.compile_model(lr=0.001)
history = GE_model.fit(epochs=30, use_early_stopping=True, early_stopping_patience=10)

predictions = GE_model.predict(x_test)
loss_history = GE_model.return_loss()

# make plots
fig, ax = plt.subplots(1, 3, figsize=(10, 3))

ax[0].plot(loss_history.history['loss'], color='blue')
ax[0].plot(loss_history.history['val_loss'], color='orange')
ax[0].set_title('Model loss', fontsize=12)
ax[0].set_ylabel('loss', fontsize=12)
ax[0].set_xlabel('epoch', fontsize=12)
ax[0].legend(['train', 'validation'])

ax[1].scatter(predictions,y_test, s=0.05, alpha=0.05, color='black')
ax[1].set_ylabel('observations')
ax[1].set_xlabel('predictions')

# get ge nonlinear function
GE_nonlinearity = GE_model.ge_nonlinearity(x_test)

ax[2].plot(GE_nonlinearity[1], GE_nonlinearity[0], color='black')
ax[2].scatter(GE_nonlinearity[2], y_test, color='gray', s=2, alpha=0.1)

ax[2].set_ylabel('observations')
ax[2].set_xlabel('latent trait ($\phi$)')

plt.tight_layout()
