from optimization_domains import *
import random
import itertools

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
    while prev_fit - fit[0] >= stop_thresh and i < max_iter:
        prev_fit = fit[0]

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
                    neighbor_dir[s][p] = np.random.choice([-1, 0, 1])  # -1 for minus, 0 for stay, 1 for plus

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
            print("Fit: " + str(temp_fit[0]))
            if len(temp_fit) > 1:
                print("Accu: " + str(temp_fit[1]))
            if temp_fit[0] < fit[0]:
                fit[0] = temp_fit[0]
                if len(temp_fit) > 1:
                    fit[1] = temp_fit[1]
                best_neighbor = n
        flat_params = neighbors[n]
        i += 1
        print("Iter " + str(i) + ": \n" + "Fit: " + str(round(fit[0], 6)) + "\n" + "prev_fit - fit: " + str(prev_fit - fit[0]) + "\n")

        # Record results
        performance.append(round(fit[0], 6))
        if len(temp_fit) > 1:
            performance_accu.append(fit[1])

    return fit

def sim_annealing(opt_params, eval_func, step_size, t_start, final_temp, max_iter, temp_schedule='exp', t_alpha=0.9995, t_delt = 0.005):
    # This function performs simulated annealing to optimize a set of given parameters.
    # opt_param: parameters to optimize. These should already be initialized to fit the optimization domain
    #   - !!!Passed in as list of np arrays!!!
    # eval_func: function to evaluate the parameters with
    # step_size: defines how "close" the neighborhood will be to a point
    # t_start: starting temperature
    # final_temp: algorithm halts when final temp is met
    # max_iter: maximum number of iterations of the algorithm
    # temp_schedule: choice of temperature schedule; can be exponential or linear
    # t_alpha: for the exponential temperature schedule, defines the rate of decrease
    # t_delt: for the linear temperature schedule, defines the rate of decrease


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

    # Simulated Annealing Loop
    i = 0
    T = t_start
    prev_fit = float("inf")
    fit = eval_func(opt_params)
    while i < max_iter and T > final_temp:
        prev_fit = fit[0]

        # Pick random neighbor
        neighbor = np.zeros(flat_params.shape[0])
        neighbor_dir = np.zeros(flat_params.shape[0], int)
        no_move = np.zeros(flat_params.shape[0])
        for p in range(flat_params.shape[0]):
            neighbor_dir[p] = np.random.choice([-1, 0, 1])  # -1 for minus, 0 for stay, 1 for plus
        neighbor = flat_params + (step_size * neighbor_dir)

        # Evaluate neighbor
        run_neighbor = opt_params

        # Expand flattened neighbor
        start = 0
        end = 0
        for j in range(len(opt_params)):
            end += len(opt_params[j].flatten())
            run_neighbor[j] = np.reshape(neighbor[start:end], (opt_params[j].shape))
            start = end

        # Run neighbor
        temp_fit = eval_func(run_neighbor)
        print("Fit: " + str(temp_fit[0]))
        if len(temp_fit) > 1:
            print("Accu: " + str(temp_fit[1]))

        # Choose neighbor or not based on annealing equation
        if temp_fit[0] < fit[0]:
            fit[0] = temp_fit[0]
            if len(temp_fit) > 1:
                fit[1] = temp_fit[1]
            flat_params = neighbor
        else:
            # Annealing equation here
            p = np.exp((fit[0] - temp_fit[0])/T)
            choose = np.random.choice([True, False], p=[p, 1-p])
            if choose:
                flat_params = neighbor
                fit[0] = temp_fit[0]
                if len(temp_fit) > 1:
                    fit[1] = temp_fit[1]
        temp.append(T)
        if temp_schedule == 'exp':
            T *= t_alpha
        elif temp_schedule == 'linear':
            T -= t_delt
        i += 1

        print('\n')
        print("T: " + str(T))
        print("i: " + str(i))
        print('\n')

        print("\n\n\n" + str(flat_params.max()) + "\n\n\n")

        # Record results
        performance.append(fit[0])
        if len(fit) > 1:
             performance_accu.append(fit[1])

    return fit


def crossover(gene0, gene1, gene_len, cross_method='point', crosspoint=None):
    # cross_method: method of crossover; can be 'point' or 'uniform'

    if crosspoint == None:
        crosspoint = gene_len/2

    gene_crossTemp0 = np.zeros(gene_len)
    gene_crossTemp1 = np.zeros(gene_len)

    # Flatten for crossover
    k = 0
    for j in range(len(gene0)):
        kEnd = k + gene0[j].flatten().shape[0]
        gene_crossTemp0[k:kEnd] = gene0[j].flatten()
        gene_crossTemp1[k:kEnd] = gene1[j].flatten()
        k = kEnd

    # Perform Crossover
    if cross_method == 'point':
        gene_cross0 = np.copy(gene_crossTemp0)
        gene_cross1 = np.copy(gene_crossTemp1)
        gene_cross0[0:crosspoint] = gene_crossTemp1[0:crosspoint]
        gene_cross1[0:crosspoint] = gene_crossTemp0[0:crosspoint]

    # Expand
    new_gene0 = np.copy(gene0)
    new_gene1 = np.copy(gene1)
    start = 0
    end = 0
    for j in range(len(gene0)):
        end += len(gene0[j].flatten())
        new_gene0[j] = np.reshape(gene_cross0[start:end], (new_gene0[j].shape))
        new_gene1[j] = np.reshape(gene_cross1[start:end], (new_gene1[j].shape))
        start = end

    return new_gene0, new_gene1

def genetic_algo(opt_params, eval_func, pop_size, bounds, div_size, stop_thresh, max_iter, seed=77):
    # This function uses a genetic algorith to optiize the given set of parameters with respect to the given evaluation function.
    # opt_param: parameters to optimize. These should already be initialized to fit the optimization domain
    #   - !!!Passed in as list of np arrays!!!
    # eval_func: function to evaluate the parameters with
    # pop_size: size of the population... must be divisible by 2
    # bounds: upper and lower bounds of the parameters... of the form [lower, upper]
    # div_size: size of the gap between possible parameter values
    # stop_thresh: halt algorithm when best 'gene' improvement is less than this value
    # max_iter: maximum number of iterations before forced stop

    if pop_size % 2 != 0:
        print("\n\n*********POPULATION SIZE MUST BE DIVISIBLE BY TWO*********\n\n")
        return

    #np.random.seed(seed)

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

    # Define possible parameter values
    param_domain = np.arange(bounds[0], bounds[1], div_size)

    # Generate population
    population = np.zeros((pop_size, flat_params.shape[0]))
    for i in range(pop_size):
        for j in range(flat_params.shape[0]):
            population[i][j] = np.random.choice(param_domain)

    # Resize population parameters to run
    population_run = [[] for i in range(pop_size)]
    gene_run = np.copy(opt_params)
    for g in range(population.shape[0]):
        start = 0
        end = 0
        for j in range(len(opt_params)):
            end += len(opt_params[j].flatten())
            gene_run[j] = np.reshape(population[g][start:end], (opt_params[j].shape))
            start = end
        population_run[g] = np.copy(gene_run)

    # Get fitness values of population
    population_fitness = np.zeros(pop_size)
    for i in range(pop_size):
        population_fitness[i] = eval_func(population_run[i])[0]

    # Sort genes based on fitness
    population_run_sort = np.copy(population_run)
    sort_order = population_fitness.argsort()
    for i in range(sort_order.shape[0]):
        population_run[i] = np.copy(population_run_sort[sort_order[i]])
    population_fitness.sort()

    # Breeding loop
    iteration = 0
    fittest = np.copy(np.min(population_fitness))
    prev_fittest = float("inf")
    while(abs(fittest - prev_fittest) > stop_thresh or iteration < max_iter):
        prev_fittest = np.copy(fittest)

        # Take most fit half
        pop_nextGen = [[] for i in range(pop_size)]
        for i in range(pop_size/2):
            pop_nextGen[i] = np.copy(population_run[i])

        # Generate new genes
        combos = [c for c in itertools.combinations(range(pop_size/2), 2)]
        breed = np.random.choice(np.arange(len(combos)), pop_size/2, replace="False")
        for i in range(pop_size/2):
            g0, g1 = crossover(population_run[combos[breed[i]][0]], population_run[combos[breed[i]][1]], total)
            pick = np.random.choice([0, 1])
            if pick == 0:
                pop_nextGen[i + (pop_size/2)] = np.copy(g0)
            elif pick == 1:
                pop_nextGen[i + (pop_size / 2)] = np.copy(g1)

        # Get fitness of new genes
        for i in range(pop_size/2, len(pop_nextGen)):
            population_fitness[i] = eval_func(pop_nextGen[i])[0]
            population_run[i] = pop_nextGen[i][:]

        # Sort genes based on fitness
        population_run_sort = np.copy(population_run)
        sort_order = population_fitness.argsort()
        for i in range(pop_size):
            population_run[i] = np.copy(population_run_sort[sort_order[i]])
        population_fitness.sort()


        # Get fittest gene and score
        fittest = np.copy(population_fitness[0])
        fittest_gene = np.copy(population_run[0])

        # # Print population status
        results = eval_func(fittest_gene)
        print("\n\n\n\n")
        print("Iteration: " + str(iteration))
        print("Eval: " + str(results[0]))
        if len(results) > 1:
            print("Accu: " + str(results[1]))

        # Record results
        if len(results) == 1:
            performance.append(results[0])
        elif len(results) > 1:
            performance_accu.append(results[1])

        iteration += 1



    return

performance = []    # holds function evaluations
performance_accu = [] # holds accuracy of an iteration if applicable
