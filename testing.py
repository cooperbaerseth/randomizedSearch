import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
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


# Generate toy data
mean1 = [-4, -4]
cov1 = [[1, 0], [0, 1]]
c1 = np.random.multivariate_normal(mean1, cov1, 5000)

mean2 = [4, 4]
cov2 = [[1, 0], [0, 1]]
c2 = np.random.multivariate_normal(mean2, cov2, 5000)

plt.plot(c1[:][:,0], c1[:][:,1], 'x')
plt.plot(c2[:][:,0], c2[:][:,1], 'x')
plt.axis('equal')
#plt.show()

# Create dataset
gaus_toyData = np.zeros((c1.shape[0]+c2.shape[0], 2))
gaus_toyLabels = np.zeros(c1.shape[0]+c2.shape[0], int)
gaus_toyData[0:c1.shape[0]] = c1
gaus_toyLabels[0:c1.shape[0]] = 0
gaus_toyData[c1.shape[0]:] = c2
gaus_toyLabels[c1.shape[0]:] = 1

# Shuffle data
s = np.arange(gaus_toyData.shape[0])
np.random.shuffle(s)
gaus_toyData = gaus_toyData[s]
gaus_toyLabels = gaus_toyLabels[s]

# Split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(gaus_toyData, gaus_toyLabels, test_size=0.33, random_state=42)
Y_train = np_utils.to_categorical(Y_train, 2)
Y_test = np_utils.to_categorical(Y_train, 2)

# Declare Keras model
neuralNet = Sequential()
neuralNet.add(Dense(1, activation='sigmoid', input_dim=X_train.shape[1]))
neuralNet.add(Dense(2, activation='softmax'))
neuralNet.compile(SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
neuralNet.summary()

# Train and evaluate model
neuralNet.fit(X_train, Y_train, epochs=20, batch_size=128)
score = neuralNet.evaluate(X_train, Y_train, batch_size=128)

print('Test score:', score[0])
print('Test accuracy:', score[1])






raw_input('Press Enter to exit')

