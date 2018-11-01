import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend
from keras.utils import np_utils
from keras.datasets import mnist
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

class modelPackage:
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test


#####################
# NEURAL NETWORKS
#####################

def neuralNet_eval(weights):
    # This function will evaluate a neural net model based on some given weights

    # holds the set of possible models to evaluate
    models = []

    # append models
    models.append(mnist_neuralNet)
    models.append(boston_neuralNet)

    # find model that matches weights
    model = None
    for i in range(len(models)):
        if weights[0].shape[0] == models[i].model.input_shape[1]:
            model = models[i].model
            x_test = models[i].x_test
            y_test = models[i].y_test
    if model == None:
        print("NO MODEL MATCH FOR WEIGHTS")
        return

    # set weights and return loss
    model.set_weights(weights)

    return model.evaluate(x_test, y_test)[0]

##################
# MNIST NEURAL NET
##################

# Load mnist
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

# Prepare dataset for model
X_train_mnist = x_train_mnist.reshape((x_train_mnist.shape[0], pow(x_train_mnist.shape[1], 2)))
X_test_mnist = x_test_mnist.reshape((x_test_mnist.shape[0], pow(x_test_mnist.shape[1], 2)))
X_train_mnist = X_train_mnist.astype('float32')
X_test_mnist = X_test_mnist.astype('float32')
X_train_mnist /= 255
X_test_mnist /= 255
Y_train_mnist = np_utils.to_categorical(y_train_mnist, 10)
Y_test_mnist = np_utils.to_categorical(y_test_mnist, 10)


# Declare Keras model
mnist_nn_model = Sequential()
mnist_nn_model.add(Dense(50, activation='sigmoid', input_dim=X_train_mnist.shape[1]))
mnist_nn_model.add(Dense(50, activation='sigmoid'))
mnist_nn_model.add(Dense(10, activation='softmax'))
mnist_nn_model.compile(SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

# Package model with test data
mnist_neuralNet = modelPackage(mnist_nn_model, X_test_mnist, Y_test_mnist)

# Neural Net Initial Weights
mnist_init_weights = mnist_nn_model.get_weights()

# weights = mnist_nerualNet.get_weights()
# in_shape = mnist_nerualNet.input_shape
# mnist_nerualNet.summary()

##############################
# BOSTON HOUSING PRICE DATASET
##############################

# Load mnist
(X_train_boston, Y_train_boston), (X_test_boston, Y_test_boston) = boston_housing.load_data()

# Declare Keras model
boston_nn_model = Sequential()
boston_nn_model.add(Dense(X_train_boston.shape[1], activation='sigmoid', input_dim=X_train_boston.shape[1]))
boston_nn_model.add(Dense(1, activation='sigmoid'))
boston_nn_model.compile(SGD(), loss='mean_squared_error', metrics=['accuracy'])

# Package model with test data
boston_neuralNet = modelPackage(boston_nn_model, X_test_boston, Y_test_boston)

# Neural Net Initial Weights
boston_init_weights = boston_nn_model.get_weights()

# weights = boston_nerualNet.get_weights()
# in_shape = boston_nerualNet.input_shape
# boston_nerualNet.summary()


#raw_input('Press Enter to exit')