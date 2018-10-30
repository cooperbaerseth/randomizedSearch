from optimization_domains import *


def hillclimb(opt_params, eval_func, sample_size, step_size, stop_thresh, max_iter):
    # This function performs randomized hillclimbing to optimize a set of given parameters.
    # opt_param: parameters to optimize. These should already be initialized to fit the optimization domain
    # eval_func: function to evaluate the parameters with
    # step_size: defines how "close" the neighborhood will be to a point
    # stop_thresh: algorithm stops if improvement is less than this value
    # max_iter: maximum number of iterations of the algorithm

    param_shape = opt_params.shape      # save original parameter shape
    opt_params = opt_params.flatten()

    i = 0
    prev_fit = float("inf")
    fit = eval_func(opt_params)
    while prev_fit - fit >= stop_thresh and i < max_iter:
        prev_fit = fit

        # get neighbor sample here
        neighbors = np.zeros((sample_size, opt_params.shape[0]))

        for j in range(neighbors.shape[0]):
            temp_fit = eval_func(neighbors[j])
            if temp_fit < fit:
                fit = temp_fit
    return fit


hillclimb(mnist_init_weights, neuralNet_eval, 50, 0.005, 0.01, 100)

