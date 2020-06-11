import mavenn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mavenn.src.utils import ge_plots_for_mavenn_demo

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
loss_history = GE_model.return_loss()

ge_plots_for_mavenn_demo(loss_history, predictions, y_test, x_test, GE_model)


