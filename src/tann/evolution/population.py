import numpy as np
from numba import njit, prange, config
from ..network.utils import get_total_parameters, compute_layer_indices
from .individual import initialize_individual, predict_individual, individual_actions


config.NUMBA_NUM_THREADS = 14  # Set number of threads for parallel execution

@njit(fastmath=True, cache=False, parallel=True)
def initialize_population(pop_size, layer_sizes, seed=None, init_method="he"):
    """Initialize a population of neural networks using individual initialization.
    
    Args:
        pop_size: Number of individuals in population
        layer_sizes: List of layer sizes
        seed: Random seed for reproducibility
        init_method: Initialization method - "he", "xavier", or "normal"
    """
    total_parameters = get_total_parameters(layer_sizes)
    population = np.zeros((pop_size, total_parameters), dtype=np.float64)

    # Initialize each individual using the individual.py function
    for i in prange(pop_size):
        # Use seed offset for each individual to ensure diversity
        individual_seed = None if seed is None else seed + i
        population[i] = initialize_individual(layer_sizes, seed=individual_seed, init_method=init_method)

    return population


@njit(fastmath=True, cache=False, parallel=False)
def predict_population(population, layer_sizes, activations, inputs,
                       population_states, prev_time):
    """
    Predict outputs for entire population with temporal memory.
    """

    pop_size = population.shape[0]
    output_size = layer_sizes[-1]

    # Pre-compute indices once for entire population
    param_indices, neuron_indices = compute_layer_indices(layer_sizes)

    # Pre-allocate outputs
    outputs = np.zeros((pop_size, output_size), dtype=np.float64)
    updated_states = np.empty_like(population_states)  # No initialization - we overwrite everything

    # Process each individual
    for i in range(pop_size):
        individual_params = population[i]
        individual_prev_states = population_states[i]

        # Predict with memory using pre-computed indices
        output, new_states = predict_individual(
            individual_params, layer_sizes, activations, inputs,
            individual_prev_states, prev_time, param_indices, neuron_indices
        )

        outputs[i] = output
        updated_states[i] = new_states

    return outputs, updated_states


@njit(fastmath=True, cache=True, parallel=True)
def get_population_actions(population, layer_sizes, layer_activations,
                       population_states, previous_time, param_indices,
                       neuron_indices, timestamps, feature_values):
    """
    Compute actions for an entire population across multiple timestamps.
    """
    pop_size = population.shape[0]
    n_timestamps = len(timestamps)
    actions = np.zeros((pop_size, n_timestamps), dtype=np.int64)

    # Process each individual in parallel
    for i in prange(pop_size):
        # Use individual_actions directly to avoid code duplication
        actions[i] = individual_actions(
            population[i], layer_sizes, layer_activations,
            population_states[i], previous_time, param_indices,
            neuron_indices, timestamps, feature_values
        )

    return actions