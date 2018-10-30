import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} )
# sess = tf.Session(config=config)
# backend.set_session(sess)
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


# Load mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

X_train = x_train.reshape((x_train.shape[0], pow(x_train.shape[1], 2)))
X_test = x_test.reshape((x_test.shape[0], pow(x_test.shape[1], 2)))
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


# Declare Keras model
neuralNet = Sequential()
neuralNet.add(Dense(50, activation='sigmoid', input_dim=X_train.shape[1]))
neuralNet.add(Dense(50, activation='sigmoid'))
neuralNet.add(Dense(10, activation='softmax'))
neuralNet.compile(SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
neuralNet.summary()

# Train and evaluate model
neuralNet.fit(X_train, Y_train, epochs=20, batch_size=128)
score = neuralNet.evaluate(X_test, Y_test, batch_size=1280)

print('Test score:', score[0])
print('Test accuracy:', score[1])






raw_input('Press Enter to exit')

