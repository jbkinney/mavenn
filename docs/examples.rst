.. _examples:

Examples
========

As described in :ref:`quickstart`, global epistasis and noise agnostic models can 
be fit to data, as shown in Tareen and Kinney (2020) using the function ``mavenn.demo``. 
Here we describe each of these analyses, as well as the snippets of code used to 
generate them. All snippets shown below are designed for use within a Jupyter Notebook, 
and assume that the following header cell has already been run. ::

    # standard imports
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    # displays logos inline within the notebook;
    # remove if using a python interpreter instead
    %matplotlib inline

    # mavenn import
    import mavenn

Additive GE model: (MPSA)
-------------------------

Example snippet for fitting additive GE model to MPSA data ::

    # load data
    mpsa_df = pd.read_csv(mavenn.__path__[0]+'/examples/datafiles/mpsa/psi_9nt_mavenn.csv')
    mpsa_df = mpsa_df.dropna()
    mpsa_df = mpsa_df[mpsa_df['values'] > 0]  # No pseudocounts

    # split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(mpsa_df['sequence'].values, np.log10(mpsa_df['values'].values))

    # load mavenn's GE model
    GE_model = mavenn.GlobalEpistasisModel(X=x_train, y=y_train, model_type='additive', alphabet_dict='rna')
    model = GE_model.define_model()
    GE_model.compile_model(lr=0.005)
    history = GE_model.fit(epochs=200, use_early_stopping=True, early_stopping_patience=20)

    # make predictions on held out test set
    predictions = GE_model.predict(x_test)
    loss_history = GE_model.return_loss()

    # plot results using helper function
    ge_plots_for_mavenn_demo(loss_history, predictions, y_test, x_test, GE_model)
	
.. image:: _static/examples_images/GE_additive_mpsa_demo.png	
	

Pairwise GE model: (MPSA)
-------------------------

Example snippet for fitting pairwise GE model to MPSA data ::

    # load data
    mpsa_df = pd.read_csv(mavenn.__path__[0]+'/examples/datafiles/mpsa/psi_9nt_mavenn.csv')
    mpsa_df = mpsa_df.dropna()
    mpsa_df = mpsa_df[mpsa_df['values'] > 0]  # No pseudocounts

    # split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(mpsa_df['sequence'].values, np.log10(mpsa_df['values'].values))

    # load mavenn's GE model
    GE_model = mavenn.GlobalEpistasisModel(X=x_train, y=y_train, model_type='pairwise',alphabet_dict='rna')
    model = GE_model.define_model()
    GE_model.compile_model(lr=0.001)
    history = GE_model.fit(epochs=200, use_early_stopping=True, early_stopping_patience=20, verbose=1)

    # make predictions on held out test set
    predictions = GE_model.predict(x_test)
    loss_history = GE_model.return_loss()

    # plot results using helper function
    ge_plots_for_mavenn_demo(loss_history, predictions, y_test, x_test, GE_model)

.. image:: _static/examples_images/GE_pairwise_mpsa_demo.png


Additive NA model: (Sort-Seq)
-----------------------------

Example snippet for inferring NA model from Sort-Seq data ::

    # load data
    sequences = np.loadtxt(mavenn.__path__[0]+'/examples/datafiles/sort_seq/full-wt/rnap_sequences.txt', dtype='str')
    bin_number = np.loadtxt(mavenn.__path__[0]+'/examples/datafiles/sort_seq/full-wt/bin_numbers.txt')

    # load mavenn's NA model
    NA_model = mavenn.NoiseAgnosticModel(X=sequences, y=bin_number)
    model = NA_model.define_model()
    NA_model.compile_model(lr=0.005)
    history = NA_model.fit(epochs=50, use_early_stopping=True, early_stopping_patience=10, verbose=1)

    # evaluate the inferred noise model for a given input range
    phi_range = np.linspace(-20,20,1000)
    noise_model = NA_model.noise_model(input_range=phi_range)

    # plot results using helper function
    na_plots_for_mavenn_demo(history, model, noise_model, phi_range)

.. image:: _static/examples_images/NA_additive_sort_seq_demo.png