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

Fill text . ::

	# load data
	mpsa_df = pd.read_csv(mavenn.__path__[0]+'/examples/datafiles/mpsa/psi_9nt_mavenn.csv')
	mpsa_df = mpsa_df.dropna()
	mpsa_df = mpsa_df[mpsa_df['values'] > 2]  # No pseudocounts

	x_train, x_test, y_train, y_test = train_test_split(mpsa_df['sequence'].values, np.log10(mpsa_df['values'].values))
	train_df = pd.DataFrame({'sequence': x_train, 'values': y_train}, columns=['sequence', 'values'])

	# load mavenn's GE model
	GE_model = mavenn.GlobalEpistasisModel(df=train_df, model_type='GE',alphabet_dict='rna')
	model = GE_model.define_model()
	GE_model.compile_model(lr=0.0025)
	history = GE_model.fit(epochs=200, use_early_stopping=True, early_stopping_patience=10)

	predictions = GE_model.predict(x_test)
	loss_history =  GE_model.return_loss()

	# make plots
	fig, ax = plt.subplots(1, 3, figsize=(10, 3))

	ax[0].plot(loss_history.history['loss'], color='blue')
	ax[0].plot(loss_history.history['val_loss'], color='orange')
	ax[0].set_title('Model loss', fontsize=12)
	ax[0].set_ylabel('loss', fontsize=12)
	ax[0].set_xlabel('epoch', fontsize=12)
	ax[0].legend(['train', 'validation'])

	ax[1].scatter(predictions,y_test,s=5, alpha=0.75, color='black')
	ax[1].set_ylabel('log$_{10}$(PSI) (test)')
	ax[1].set_xlabel('predictions (test)')

	phi = np.arange(-5,3,0.1)
	ax[2].plot(phi,GE_model.ge_nonlinearity(phi), color='black')
	ax[2].set_ylabel('log$_{10}$(PSI) (test)')
	ax[2].set_xlabel('latent trait ($\phi$)')
	
.. image:: _static/examples_images/GE_additive_mpsa_demo.png	
	
