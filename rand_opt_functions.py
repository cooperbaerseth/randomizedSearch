from optimization_domains import *


def hillclimb(opt_params, eval_func, step_size, max_iter):
    # This function performs randomized hillclimbing to optimize a set of given parameters.
    # opt_param: parameters to optimize. These should already be initialized to fit the optimization domain
    # eval_func: function to evaluate the parameters with
    # step_size: defines how "close" the neighborhood will be to a point
    # max_iter: maximum number of iterations of the algorithm

    fitness = eval_func(opt_params)


    return


hillclimb(mnist_init_weights, neuralNet_eval, 0.005, 100)


