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

    x_train, x_test, y_train, y_test = train_test_split(mpsa_df['sequence'].values, np.log10(mpsa_df['values'].values))
    train_df = pd.DataFrame({'sequence': x_train, 'values': y_train}, columns=['sequence', 'values'])

    # load mavenn's GE model
    GE_model = mavenn.GlobalEpistasisModel(df=train_df, model_type='additive', alphabet_dict='rna')
    model = GE_model.define_model()
    GE_model.compile_model(lr=0.005)
    history = GE_model.fit(epochs=200, use_early_stopping=True, early_stopping_patience=20)

    predictions = GE_model.predict(x_test)
    loss_history =  GE_model.return_loss()

    # make plots
    ge_plots_for_mavenn_demo(loss_history, predictions, y_test, x_test, GE_model)
	
.. image:: _static/examples_images/GE_additive_mpsa_demo.png	
	

Pairwise GE model: (MPSA)
-------------------------

Example snippet for fitting pairwise GE model to MPSA data ::

    # load data
    mpsa_df = pd.read_csv(mavenn.__path__[0]+'/examples/datafiles/mpsa/psi_9nt_mavenn.csv')
    mpsa_df = mpsa_df.dropna()
    mpsa_df = mpsa_df[mpsa_df['values'] > 0]  # No pseudocounts

    x_train, x_test, y_train, y_test = train_test_split(mpsa_df['sequence'].values, np.log10(mpsa_df['values'].values))
    train_df = pd.DataFrame({'sequence': x_train, 'values': y_train}, columns=['sequence', 'values'])

    # load mavenn's GE model
    GE_model = mavenn.GlobalEpistasisModel(df=train_df, model_type='pairwise',alphabet_dict='rna')
    model = GE_model.define_model()
    GE_model.compile_model(lr=0.001)
    history = GE_model.fit(epochs=200, use_early_stopping=True, early_stopping_patience=20, verbose=1)

    predictions = GE_model.predict(x_test)
    loss_history =  GE_model.return_loss()

    # make plots
    ge_plots_for_mavenn_demo(loss_history, predictions, y_test, x_test, GE_model)

.. image:: _static/examples_images/GE_pairwise_mpsa_demo.png