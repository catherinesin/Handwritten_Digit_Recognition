import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Multilayer Perceptron Model for MNIST dataset
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import layers
from keras.utils import np_utils
import numpy as np
import time
import datetime

# Define paramaters for the model
learning_rate = 0.1
batch_size = 1000
n_epochs = 30

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape input size, convert the 28*28 vector into a 1-d vector of 1*784
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

# Normalization
x_train = x_train / 255
x_test = x_test / 255

# One hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
# Add activation layers
# Loss function for one-hot vector
model = models.Sequential()
model.add(layers.Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(num_classes, kernel_initializer='normal', activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start_time = time.time()

# TensorBoard writer
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir_train = 'logs/logreg/' + current_time + '/train_b'
tf_callback_train = keras.callbacks.TensorBoard(log_dir = log_dir_train)

# Fit the model
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = n_epochs, 
          batch_size = batch_size, verbose = 2, callbacks = [tf_callback_train])

# Evaluation
results = model.evaluate(x_test, y_test, verbose=0)
print("loss and accuracy: ", results)

end_time = time.time()
print('Total Time {} ms.'.format(end_time - start_time))
model.summary()