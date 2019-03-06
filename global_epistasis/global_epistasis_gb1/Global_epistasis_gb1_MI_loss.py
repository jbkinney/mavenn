
# coding: utf-8

# ### Import libraries and read data

# In[1]:


import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#input_data_ordered_raw =  pd.read_csv('rnap_scanned_over_ecoli_genome200k.csv')
input_data_ordered_raw = pd.read_csv('../GB1.csv')
#sequences = input_data_ordered_raw['seq'].tolist()


# In[2]:


input_data_ordered_raw.head()


# In[3]:


sequences = input_data_ordered_raw['seq'][0:20000]
val = input_data_ordered_raw['val'][0:20000]


# In[4]:


val_norm = (val-min(val))/(max(val)-min(val))
#plt.hist(val_norm,bins=100)
#plt.show()


# In[5]:


len(sequences)


# In[6]:


np.random.seed(0)


# In[7]:


T = 300
t_max = 15
t_bg = 0.01

size_of_data = 10000
#size_of_data = len(sequences)

input_data_ordered_raw_copy = input_data_ordered_raw[0:size_of_data].copy()

#temp = np.exp(-(np.array(input_data_ordered_raw['val'][0:size_of_data]))/T)/(1+np.exp(-(np.array(input_data_ordered_raw['val'][0:size_of_data]))/T))
#plt.hist(t_max*temp+t_bg,bins=100)
#plt.show()
#np.array(input_data_ordered_raw['val'][0:100000])


# ### Add transcription column to dataframe

# In[8]:


#input_data_ordered_raw_copy['t'] = t_max*temp+t_bg


# In[9]:


input_data_ordered_raw_copy.head()


# ### Draw read counts according to poission distribution and add to dataframe

# In[10]:


input_data_ordered_raw_copy.head(10)


# ## Split the data into test and train

# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


#x_train, x_test, y_train, y_test = train_test_split(input_data_ordered_raw_copy['seq'],input_data_ordered_raw_copy['t_norm'],test_size=0.2,random_state=4)
#x_train, x_test, y_train, y_test = train_test_split(input_data_ordered_raw_copy['seq'],input_data_ordered_raw_copy['C'],test_size=0.2)
#x_train, x_test, y_train, y_test = train_test_split(input_data_ordered_raw_copy['seq'],input_data_ordered_raw_copy['val_norm'],test_size=0.2)
x_train, x_test, y_train, y_test = train_test_split(sequences,val_norm,test_size=0.2)



# In[13]:

'''
plt.hist(y_train,bins=100,color='r',density=True,alpha=0.5)
plt.hist(y_test,bins=100,color='b',density=True,alpha=0.5)
plt.show()
'''

# ## One-hot encode the data ... this takes a few minutes

# In[14]:


temp_x_train = []
temp_x_test = []

for reshape_index in range(len(x_train)):
    temp_x_train.append(list(np.array(x_train)[reshape_index]))

for reshape_test_index in range(len(x_test)):
    temp_x_test.append(list(np.array(x_test)[reshape_test_index]))
    
x_train = temp_x_train
x_test = temp_x_test

x_train = np.array(x_train)
x_test = np.array(x_test)


# In[15]:


x_train[0]


# In[16]:


base_dict = {"K":0, "R":1, "H":2, "E":3, "D":4, "N":5, "Q":6, "T":7, "S":8, "C":9, "G":10, "A":11, "V":12, "L":13, "I":14, "M":15, "P":16, "Y":17, "F":18, "W":19}


# In[17]:


test_size = len(y_test)


# In[18]:


x_train_tensor = np.zeros(list(x_train.shape) + [20])    # shape: (batch_size, 4)
x_test_tensor = np.zeros(list(x_test.shape) + [20])    # shape: (batch_size, 4)
base_dict = {"K":0, "R":1, "H":2, "E":3, "D":4, "N":5, "Q":6, "T":7, "S":8, "C":9, "G":10, "A":11, "V":12, "L":13, "I":14, "M":15, "P":16, "Y":17, "F":18, "W":19}

#base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
num_sample = len(x_train_tensor)
seq_length = len(x_train[0])
test_size = len(x_test)


#naive one-hot encoding
for row in range(num_sample):
    for col in range(seq_length):
        x_train_tensor[row,col,base_dict[x_train[row,col]]] = 1
        if(row<test_size):
            x_test_tensor[row,col,base_dict[x_test[row,col]]] = 1


# In[19]:


print('Training set shape: {}'.format(x_train_tensor.shape))
print('Training set label shape: {}'.format(y_train.shape))

print('Test set shape: {}'.format(x_test_tensor.shape))
print('Test set label shape: {}'.format(y_test.shape))


# In[20]:


#REGG
x_train_tensor[0][3]


# In[41]:


y_train.shape


# In[42]:


y_train = np.array(y_train).reshape(y_train.shape[0],1)


# ## Custom error metric

# In[43]:


# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))


# ## Custom Loss

# In[44]:


import keras.backend as K
import tensorflow as tf

K.clear_session()

def kullback_leibler_divergence_ammar(y_true, y_pred):
    tf.print(y_true,'tui')
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


# In[55]:


# Calculate the entropy of a 1D tensor, fuzzing the edges with epsilon to keep numbers
# clean.
def calculate_entropy(y, epsilon):
    clipped = tf.clip_by_value(y, epsilon, 1 - epsilon)
    return -tf.cast(tf.reduce_sum(clipped * tf.log(clipped)), dtype=tf.float32)


# Sandbox for developing calculating the entropies of y
def tf_entropies(y_true,y_pred, epsilon, nbins):
    # Create histograms for the activations in the batch.
    value_range = [0.0, 1.0]
    # For prototype, only consider first two features.
    neuron1 = y_true
    neuron2 = y_pred
    hist1 = tf.histogram_fixed_width(neuron1, value_range, nbins=nbins)
    hist2 = tf.histogram_fixed_width(neuron2, value_range, nbins=nbins)
    # Normalize
    count = tf.cast(tf.count_nonzero(hist1), tf.int32)
    dist1 = tf.divide(hist1, count)
    dist2 = tf.divide(hist2, count)
    neuron1_entropy = calculate_entropy(dist1, epsilon)
    neuron2_entropy = calculate_entropy(dist2, epsilon)


    # Calculate the joint distribution and then get the entropy
    recast_n1 = tf.cast(tf.divide(tf.cast(nbins * neuron1, tf.int32), nbins), tf.float32)
    meshed = recast_n1 + tf.divide(neuron2, nbins)  # Shift over the numbers for neuron2
    joint_hist = tf.histogram_fixed_width(meshed, value_range, nbins=nbins * nbins)
    joint_dist = tf.divide(joint_hist, count)
    joint_entropy = calculate_entropy(joint_dist, epsilon)
    #print('xXXXxxxXXXX')
    #print(neuron1_entropy+neuron2_entropy-joint_entropy)
    #return neuron1_entropy,neuron2_entropy,joint_entropy
    return dist1,dist2,joint_dist

def mi_loss(y_true,y_pred):
    epsilon = 1e-5
    bins = 10
    return tf_entropies(y_true,y_pred,epsilon,bins)

from keras.layers import multiply

def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    dist1, dist2, joint_dist = tf_entropies(y_true,y_pred,1e-10,1000)
    #return K.sum(y_true * K.log(y_true / y_pred), axis=-1)
    return K.sum(joint_dist * K.log(joint_dist/  (K.dot(dist1,dist2))), axis=-1)

def mean_squared_error(y_true, y_pred):
    print("XXXXXXX: ",y_true)
    return K.mean(K.square(y_pred - y_true), axis=-1)

# In[56]:


from keras.models import Sequential 
from keras.layers import Dense, Activation, Flatten, Dropout 
from keras import regularizers
from keras.optimizers import SGD, Adam, RMSprop

#output_dim = nb_classes = 15 
#input_dim = seq_length
model = Sequential() 
model.add(Flatten())
#model.add(Dense(100, activation='relu',input_shape=(41,4)))
#model.add(Dropout(0.25))
#model.add(Dense(41, activation='relu'))
#model.add(Dropout(0.25))
#model.add(Dense(4, activation='relu'))
#model.add(Dropout(0.25))
#model.add(Dense(20, activation='linear'))
model.add(Dense(1, activation='linear'))
#model.add(Dense(20, activation='tanh'))
model.add(Dense(20, activation='sigmoid'))
#model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='linear'))
#model.add(Dense(1, activation='tanh'))

#batch_size = 10000 
#nb_epoch = 20


# In[57]:


model.compile(loss=kullback_leibler_divergence,optimizer=Adam(lr=0.0001), metrics=['mean_absolute_error'])
#model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.0005), metrics=['mean_absolute_error'])


# In[58]:


#history = model.fit(x_train_flat, y_train, validation_split=0.2, epochs=25)  # starts training
history = model.fit(x_train_tensor, y_train, validation_split=0.2, epochs=3)  # starts training


