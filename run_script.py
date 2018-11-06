from rand_opt_functions import *
import time




'''
*******MAIN*******
'''

performance = []
performance_accu = []
temp = []


def run():
    # Hillclimb
    # NN
    #hillclimb(toy_gauss_init_weights, neuralNet_eval, 40, 0.05, 0.001, 100)
    #hillclimb(boston_init_weights, neuralNet_eval, 2000, 0.05, 0.001, 100)
    #hillclimb(mnist_init_weights, neuralNet_eval, 50, 0.05, 0.001, 100)

    # Classic Opt
    hillclimb(classic_optParams, ackley, 40, 0.04, 0.001, 100)
    #hillclimb(classic_optParams, himmelblau, 40, 0.05, 0.001, 100)


    # Simulated Annealing
    # NN
    #sim_annealing(toy_gauss_init_weights, neuralNet_eval, 0.05, 20, 2, 20000, temp_schedule='exp', t_alpha=0.995)
    #sim_annealing(boston_init_weights, neuralNet_eval, 0.05, 20, 2, 20000)
    #sim_annealing(mnist_init_weights, neuralNet_eval, 0.05, 20, 0.1, 20000, t_alpha=0.95)

    # Classic Opt
    #sim_annealing(classic_optParams, ackley, 0.05, 20, 2, 20000000, temp_schedule='linear')
    #sim_annealing(classic_optParams, himmelblau, 0.05, 20, 2, 2000, temp_schedule='exp')

    # Genetic Algorithm
    # NN
    #genetic_algo(toy_gauss_init_weights, neuralNet_eval, 100, [-2, 2], 0.05, 0.001, 200)
    #genetic_algo(boston_init_weights, neuralNet_eval, 100, [-2, 2], 0.05, 0.001, 200)
    #genetic_algo(mnist_init_weights, neuralNet_eval, 200, [-2, 2], 0.05, 0.001, 10)

    # Classic Opt
    #genetic_algo(classic_optParams, ackley, pop, [domain_x[0], domain_x[1]], 0.05, 0.001, 200)
    #genetic_algo(classic_optParams, himmelblau, pop, [domain_x[0], domain_x[1]], 0.05, 0.001, 200)

def plot():
    zoom = (max(performance[:]) / 50)

    if performance_accu != []:
        if temp != []:
            fig = plt.figure()
            ax = fig.add_subplot(311)
            ax.plot(range(len(performance)), performance[:])
            ax = fig.add_subplot(312)
            ax.plot(range(len(performance_accu)), performance_accu[:])
            ax = fig.add_subplot(313)
            ax.plot(range(len(temp)), temp[:])
        else:
            fig = plt.figure()
            ax = fig.add_subplot(211)
            ax.plot(range(len(performance)), performance[:])
            ax = fig.add_subplot(212)
            ax.plot(range(len(performance_accu)), performance_accu[:])
    else:
        if temp != []:
            fig = plt.figure()
            ax = fig.add_subplot(211)
            ax.plot(range(len(performance)), performance[:])
            ax = fig.add_subplot(212)
            ax.plot(range(len(temp)), temp[:])
        else:
            plt.figure()
            plt.plot(range(len(performance)), performance[:])
            plt.ylim(min(performance[:]) - zoom/5, max(performance[:]) + zoom)
            plt.xlim(-2, len(performance) + 5)

    plt.suptitle("Min Value: " + str(round(min(performance), 2)) + " Time Taken: " + str(round(t, 3)))
    plt.show()

t0 = time.time()
run()
t1 = time.time()
t = t1 - t0

plot()

# min_val = []
# for i in range(6, 501, 2):
#     performance = []
#     pop = i
#     run()
#     min_val.append(min(performance))
# zoom = (max(min_val[:]) / 50)
# plt.figure()
# plt.plot(range(len(min_val)), min_val[:])
# plt.ylim(min(min_val[:]) - zoom / 5, max(min_val[:]) + zoom)
# plt.xlim(-2, len(min_val) + 5)

#
# min_val = []
# for i in range(20, 200):
#     performance = []
#     te = i
#     run()
#     min_val.append(min(performance))
# zoom = (max(min_val[:]) / 50)
# plt.figure()
# plt.plot(range(len(min_val)), min_val[:])
# plt.ylim(min(min_val[:]) - zoom / 5, max(min_val[:]) + zoom)
# plt.xlim(-2, len(min_val) + 5)

# min_val = []
# for i in range(20, 500):
#     performance = []
#     randrest = i
#     run()
#     min_val.append(min(performance))
# zoom = (max(min_val[:]) / 50)
# plt.figure()
# plt.plot(range(len(min_val)), min_val[:])
# plt.ylim(min(min_val[:]) - zoom / 5, max(min_val[:]) + zoom)
# plt.xlim(-2, len(min_val) + 5)



plt.pause(0.05)
input('Press Enter to exit')

