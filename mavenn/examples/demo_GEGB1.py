import mavenn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mavenn.src.utils import load_olson_data_GB1

# load data using helper method in mavenn utils
gb1_df = load_olson_data_GB1()

# split data into train and test
x_train, x_test, y_train, y_test = train_test_split(gb1_df['sequence'].values, gb1_df['values'].values)

# load mavenn's GE model
GER = mavenn.Model(regression_type='GE',
                   X=x_train,
                   y=y_train,
                   model_type='additive',
                   learning_rate=0.001,
                   alphabet_dict='protein')

# fit model to data
GER.fit(epochs=200, use_early_stopping=True, early_stopping_patience=20, verbose=1)

# make predictions on held out test set
loss_history = GER.model.return_loss()
predictions = GER.model.predict(x_test)

loss_history = GER.model.return_loss()
predictions = GER.model.predict(x_test)

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
GE_nonlinearity = GER.ge_nonlinearity(x_test)

ax[2].plot(GE_nonlinearity[1], GE_nonlinearity[0], color='black')
ax[2].scatter(GE_nonlinearity[2], y_test, color='gray', s=2, alpha=0.1)

ax[2].set_ylabel('observations')
ax[2].set_xlabel('latent trait ($\phi$)')

plt.tight_layout()
