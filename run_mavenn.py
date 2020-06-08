# turn this into demo
import mavenn
import pandas as pd
from mavenn.src.utils import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# load mpsa input
mpsa_df = pd.read_csv('raw_data/mpsa/psi_9nt_mavenn.csv')
mpsa_df = mpsa_df.dropna()
mpsa_df = mpsa_df[mpsa_df['values'] > 2]  # No pseudocounts

x_train, x_test, y_train, y_test = train_test_split(mpsa_df['sequence'].values, np.log10(mpsa_df['values'].values))

train_df = pd.DataFrame({'sequence': x_train, 'values': y_train}, columns=['sequence', 'values'])

# load mavenn's GE model
GE_model = mavenn.GlobalEpistasisModel(df=train_df, model_type='GE')
model = GE_model.define_model()
GE_model.compile_model(lr=0.005)

history = GE_model.fit(epochs=100)
GE_model.plot_losses()

predictions = GE_model.predict(x_test)

plt.scatter(y_test, predictions, alpha=0.75, color='black')
plt.xlabel('Ground truth (test)')
plt.ylabel('predictions (test)')
plt.show()


