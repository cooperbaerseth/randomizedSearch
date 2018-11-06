import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend
from keras.utils import np_utils
from keras.datasets import mnist
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

#####################################
# CLASSIC OPTIMIZATION PROBLEMS
#####################################

def ackley(X):
    return np.array([-20*np.exp(-0.2*np.sqrt(0.5*(np.power(X[0], 2) + np.power(X[1], 2)))) - np.exp(0.5*(np.cos(2*np.pi*X[0]) + np.cos(2*np.pi*X[1]))) + np.e + 20])

def himmelblau(X):
    return np.array([(np.power((np.power(X[0], 2) + X[1] - 71), 2) + np.power((X[0] + np.power(X[1], 2) - 34), 2))])

# Define domain
r = 10
domain_x = [-r, r]
domain_y = [-r, r]

# Visualize optimization functions
X = np.arange(domain_x[0], domain_x[1], 0.05)
Y = np.arange(domain_y[0], domain_y[1], 0.05)
X_mesh, Y_mesh = np.meshgrid(X, Y)

Z = ackley([X_mesh, Y_mesh])[0]
Z1 = himmelblau([X_mesh, Y_mesh])[0]

# fig = plt.figure()
# ax = fig.add_subplot(211, projection='3d')
# ax.plot_surface(X_mesh, Y_mesh, Z, cmap=cm.viridis)
# ax = fig.add_subplot(212, projection='3d')
# ax.plot_surface(X_mesh, Y_mesh, Z1, cmap=cm.viridis)
# plt.show()
#
# plt.pause(0.05)
# input('Press Enter to exit')


# Initialize parameters
classic_optParams = np.random.choice(X, 2)

#####################
# NEURAL NETWORKS
#####################

class modelPackage:
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

def neuralNet_eval(weights):
    # This function will evaluate a neural net model based on some given weights

    # holds the set of possible models to evaluate
    models = []

    # append models
    models.append(mnist_neuralNet)
    models.append(boston_neuralNet)
    models.append(toy_gauss_neuralNet)

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

    return model.evaluate(x_test, y_test)

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
#mnist_nn_model.summary()

##############################
# BOSTON HOUSING PRICE DATASET
##############################

# Load Boston
(X_train_boston, Y_train_boston), (X_test_boston, Y_test_boston) = boston_housing.load_data()

# Declare Keras model
boston_nn_model = Sequential()
boston_nn_model.add(Dense(X_train_boston.shape[1], activation='sigmoid', input_dim=X_train_boston.shape[1]))
boston_nn_model.add(Dense(1, activation='sigmoid'))
boston_nn_model.compile(SGD(), loss='mean_squared_error', metrics=['accuracy'])

# Package model with test data
boston_neuralNet = modelPackage(boston_nn_model, X_test_boston, Y_test_boston)

# Neural Net Initial Weights
#boston_init_weights = boston_nn_model.get_weights()

# weights = boston_nn_model.get_weights()
# in_shape = boston_nn_model.input_shape
#boston_nn_model.summary()


######################
# TOY GAUSSIAN DATASET
######################

# Generate toy data
mean1 = [-4, -4]
cov1 = [[1, 0], [0, 1]]
c1 = np.random.multivariate_normal(mean1, cov1, 5000)

mean2 = [4, 4]
cov2 = [[1, 0], [0, 1]]
c2 = np.random.multivariate_normal(mean2, cov2, 5000)

# plt.plot(c1[:][:,0], c1[:][:,1], 'x')
# plt.plot(c2[:][:,0], c2[:][:,1], 'x')
# plt.axis('equal')
# plt.show()

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
X_train_toyGaus, X_test_toyGaus, Y_train_toyGaus, Y_test_toyGaus = train_test_split(gaus_toyData, gaus_toyLabels, test_size=0.33, random_state=42)
Y_train_toyGaus = np_utils.to_categorical(Y_train_toyGaus, 2)
Y_test_toyGaus = np_utils.to_categorical(Y_test_toyGaus, 2)

# Declare Keras model
toy_gauss_nn_model = Sequential()
toy_gauss_nn_model.add(Dense(1, activation='sigmoid', input_dim=X_train_toyGaus.shape[1]))
toy_gauss_nn_model.add(Dense(2, activation='softmax'))
toy_gauss_nn_model.compile(SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
#toy_gauss_nn_model.summary()

# Package model with test data
toy_gauss_neuralNet = modelPackage(toy_gauss_nn_model, X_test_toyGaus, Y_test_toyGaus)

# Neural Net Initial Weights
toy_gauss_init_weights = toy_gauss_nn_model.get_weights()


# raw_input('Press Enter to exit')