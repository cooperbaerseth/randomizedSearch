from optimization_domains import *
import random

def hillclimb(opt_params, eval_func, sample_size, step_size, stop_thresh, max_iter):
    # This function performs randomized hillclimbing to optimize a set of given parameters.
    # opt_param: parameters to optimize. These should already be initialized to fit the optimization domain
    #   - !!!Passed in as list of np arrays!!!
    # eval_func: function to evaluate the parameters with
    # step_size: defines how "close" the neighborhood will be to a point
    # stop_thresh: algorithm stops if improvement is less than this value
    # max_iter: maximum number of iterations of the algorithm

    # Flatten opt_params
    total = 0
    for i in range(len(opt_params)):
                total += opt_params[i].flatten().shape[0]
    flat_params = np.zeros(total)
    k = 0
    for i in range(len(opt_params)):
        kEnd = k + opt_params[i].flatten().shape[0]
        flat_params[k:kEnd] = opt_params[i].flatten()
        k = kEnd

    # Hillclimb loop
    i = 0
    prev_fit = float("inf")
    fit = eval_func(opt_params)
    while prev_fit - fit >= stop_thresh and i < max_iter:
        prev_fit = fit

        # get neighbor samples
        neighbors = np.zeros((sample_size, flat_params.shape[0]))
        neighbor_dir = np.zeros((sample_size, flat_params.shape[0]), int)
        dir_archive = []
        no_move = np.zeros(flat_params.shape[0])

        print("Generating neighbors (iter " + str(i) + ")...")
        for s in range(sample_size):
            repeat = any((neighbor_dir[s] == x).all() for x in dir_archive)
            while  repeat or (neighbor_dir[s] == no_move).all():
                for p in range(flat_params.shape[0]):
                    neighbor_dir[s][p] = np.random.choice([-1, 0, 1], 1)  # -1 for minus, 0 for stay, 1 for plus

            dir_archive.append(neighbor_dir[s])
            neighbors[s] = flat_params + (step_size*neighbor_dir[s])


        # Run neighbors
        run_neighbor = opt_params
        for n in range(neighbors.shape[0]):

            # Expand flattened neighbor
            start = 0
            end = 0
            for j in range(len(opt_params)):
                end += len(opt_params[j].flatten())
                run_neighbor[j] = np.reshape(neighbors[n][start:end], (opt_params[j].shape))
                start = end

            # Run neighbor
            temp_fit = eval_func(run_neighbor)
            print(temp_fit)
            if temp_fit < fit:
                fit = temp_fit
                best_neighbor = n
        flat_params = neighbors[n]
        i += 1
        print("Iter " + str(i) + ": \n" + "Fit: " + str(round(fit, 6)) + "\n" + "prev_fit - fit: " + str(prev_fit - fit) + "\n")
    return fit

hillclimb(boston_init_weights, neuralNet_eval, 2000, 0.05, 0.001, 100)
#hillclimb(mnist_init_weights, neuralNet_eval, 50, 0.05, 0.001, 100)

