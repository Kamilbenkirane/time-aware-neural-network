# ============================================================================
# INDIVIDUAL WEIGHT INITIALIZATION
# ============================================================================

@njit(fastmath=True, cache=True)
def initialize_weights(layer_sizes, seed=None):
    """Initialize random weights for neural network."""
    if seed is not None:
        np.random.seed(seed)

    total_parameters = get_total_parameters(layer_sizes)
    parameters = np.random.randn(total_parameters).astype(np.float64) * 0.3
    return weights


# ============================================================================
# POPULATION INITIALIZATION
# ============================================================================

@njit(fastmath=True, cache=True)
def initialize_population(population_size, layer_sizes, seed=None):
    """Initialize random weights for entire population."""
    if seed is not None:
        np.random.seed(seed)

    total_parameters = get_total_parameters(layer_sizes)
    population = np.empty((population_size, total_weights), dtype=np.float64)

    for i in range(population_size):
        # Use different seed for each individual for diversity
        individual_seed = None if seed is None else seed + i
        population[i] = initialize_weights(layer_sizes, individual_seed)

    return population


# ============================================================================
# FITNESS FUNCTIONS
# ============================================================================

@njit(fastmath=True, cache=True)
def evaluate_fitness_sphere(weights):
    """Sphere function: f(x) = -sum(x_i^2) (negative for maximization)."""
    return -np.sum(weights * weights)


@njit(fastmath=True, cache=True)
def evaluate_fitness_rastrigin(weights):
    """Rastrigin function: f(x) = -[A*n + sum(x_i^2 - A*cos(2*pi*x_i))] (negative for maximization)."""
    A = 10.0
    n = len(weights)
    sum_term = 0.0
    for i in range(n):
        sum_term += weights[i] ** 2 - A * np.cos(2 * np.pi * weights[i])
    return -(A * n + sum_term)


@njit(fastmath=True, cache=True)
def evaluate_fitness(weights, fitness_id):
    """Generic fitness dispatch: 0=sphere, 1=rastrigin, default=sphere."""
    if fitness_id == 0:
        return evaluate_fitness_sphere(weights)
    elif fitness_id == 1:
        return evaluate_fitness_rastrigin(weights)
    else:
        # Default to sphere function
        return evaluate_fitness_sphere(weights)


# ============================================================================
# POPULATION FITNESS EVALUATION
# ============================================================================

@njit(fastmath=True, cache=True, parallel=True)
def evaluate_population_fitness(population, fitness_id):
    """Evaluate fitness for entire population (parallel)."""
    population_size = population.shape[0]
    fitness_scores = np.empty(population_size, dtype=np.float64)

    for i in prange(population_size):
        fitness_scores[i] = evaluate_fitness(population[i], fitness_id)

    return fitness_scores


# ============================================================================
# TOURNAMENT SELECTION
# ============================================================================

@njit(fastmath=True, cache=True)
def tournament_selection_single(fitness_scores, tournament_size, seed=None):
    """Select one individual using tournament selection."""
    if seed is not None:
        np.random.seed(seed)

    population_size = len(fitness_scores)

    # Select first tournament participant
    best_index = np.random.randint(0, population_size)
    best_fitness = fitness_scores[best_index]

    # Compare with remaining tournament participants
    for i in range(tournament_size - 1):
        candidate_index = np.random.randint(0, population_size)
        candidate_fitness = fitness_scores[candidate_index]

        if candidate_fitness > best_fitness:  # Maximization
            best_fitness = candidate_fitness
            best_index = candidate_index

    return best_index


@njit(fastmath=True, cache=True, parallel=True)
def tournament_selection(fitness_scores, num_parents, tournament_size, seed=None):
    """Select multiple parents using tournament selection (parallel)."""
    if seed is not None:
        np.random.seed(seed)

    parent_indices = np.empty(num_parents, dtype=np.int64)

    for i in prange(num_parents):
        # Use different seed for each tournament for diversity
        parent_seed = None if seed is None else seed + i
        parent_indices[i] = tournament_selection_single(fitness_scores, tournament_size, parent_seed)

    return parent_indices


# ============================================================================
# CROSSOVER OPERATIONS
# ============================================================================

@njit(fastmath=True, cache=True, parallel=True)
def crossover_single_point(population, parent_indices, seed=None):
    """Single-point crossover for multiple parent pairs (parallel)."""
    if seed is not None:
        np.random.seed(seed)

    num_parents = len(parent_indices)
    num_pairs = num_parents // 2
    genome_length = population.shape[1]

    # Create offspring array
    offspring = np.empty((num_pairs * 2, genome_length), dtype=np.float64)

    for i in prange(num_pairs):
        # Get parent indices
        parent1_idx = parent_indices[2 * i]
        parent2_idx = parent_indices[2 * i + 1]

        # Set unique seed for this crossover
        if seed is not None:
            np.random.seed(seed + i)

        # Choose crossover point (avoid endpoints)
        crossover_point = np.random.randint(1, genome_length)

        # Create offspring indices
        offspring1_idx = 2 * i
        offspring2_idx = 2 * i + 1

        # Copy parent segments before crossover point
        for j in range(crossover_point):
            offspring[offspring1_idx, j] = population[parent1_idx, j]
            offspring[offspring2_idx, j] = population[parent2_idx, j]

        # Swap segments after crossover point
        for j in range(crossover_point, genome_length):
            offspring[offspring1_idx, j] = population[parent2_idx, j]
            offspring[offspring2_idx, j] = population[parent1_idx, j]

    return offspring


@njit(fastmath=True, cache=True, parallel=True)
def crossover_uniform(population, parent_indices, crossover_rate=0.5, seed=None):
    """Uniform crossover for multiple parent pairs (parallel)."""
    if seed is not None:
        np.random.seed(seed)

    num_parents = len(parent_indices)
    num_pairs = num_parents // 2
    genome_length = population.shape[1]

    # Create offspring array
    offspring = np.empty((num_pairs * 2, genome_length), dtype=np.float64)

    for i in prange(num_pairs):
        # Get parent indices
        parent1_idx = parent_indices[2 * i]
        parent2_idx = parent_indices[2 * i + 1]

        # Set unique seed for this crossover
        if seed is not None:
            np.random.seed(seed + i)

        # Create offspring indices
        offspring1_idx = 2 * i
        offspring2_idx = 2 * i + 1

        # Uniform crossover: for each gene, randomly choose parent
        for j in range(genome_length):
            if np.random.random() < crossover_rate:
                # Swap genes
                offspring[offspring1_idx, j] = population[parent2_idx, j]
                offspring[offspring2_idx, j] = population[parent1_idx, j]
            else:
                # Keep original
                offspring[offspring1_idx, j] = population[parent1_idx, j]
                offspring[offspring2_idx, j] = population[parent2_idx, j]

    return offspring
