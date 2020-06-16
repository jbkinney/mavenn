import mavenn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mavenn.src.utils import ge_plots_for_mavenn_demo, get_example_dataset

# load data
X, y = get_example_dataset(name='GB1-DMS')

# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y)

# load mavenn's GE model
GER = mavenn.Model(regression_type='GE',
                   X=x_train,
                   y=y_train,
                   model_type='additive',
                   learning_rate=0.001,
                   alphabet_dict='protein',
                   ohe_single_batch_size=100000)

# fit model to data
GER.fit(epochs=100, use_early_stopping=True, early_stopping_patience=20, verbose=1)

# make predictions on held out test set
loss_history = GER.model.return_loss()
predictions = GER.model.predict(x_test)

loss_history = GER.model.return_loss()
predictions = GER.model.predict(x_test)

ge_plots_for_mavenn_demo(loss_history, predictions, y_test, x_test, GER)
