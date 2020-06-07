# turn this into demo
import mavenn
import pandas as pd

# load mpsa input
mpsa_df = pd.read_csv('raw_data/mpsa/psi_9nt_mavenn.csv')
mpsa_df = mpsa_df.dropna()
mpsa_df = mpsa_df[mpsa_df['values']>0] # No pseudocounts

# load mavenn's GE model
GE_model = mavenn.GlobalEpistasisModel(df=mpsa_df, model_type='GE')
model = GE_model.define_model()
GE_model.compile_model()

history = GE_model.model_fit()


