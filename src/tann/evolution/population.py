import numpy as np
from numba import njit, prange
from ..network.utils import get_total_parameters, compute_layer_indices
from .individual import initialize_individual, predict_individual



@njit(fastmath=True, cache=False, parallel=True)
def initialize_population(pop_size, layer_sizes, seed=None):
    """Initialize a population of neural networks using individual initialization."""
    total_parameters = get_total_parameters(layer_sizes)
    population = np.zeros((pop_size, total_parameters), dtype=np.float64)

    # Initialize each individual using the individual.py function
    for i in prange(pop_size):
        # Use seed offset for each individual to ensure diversity
        individual_seed = None if seed is None else seed + i
        population[i] = initialize_individual(layer_sizes, seed=individual_seed)

    return population


@njit(fastmath=True, cache=False, parallel=False)
def predict_population(population, layer_sizes, activations, inputs,
                       population_states, population_prev_times):
    """
    Predict outputs for entire population with temporal memory.
    """
    current_time, x_vector = inputs
    pop_size = population.shape[0]
    output_size = layer_sizes[-1]

    # Pre-compute indices once for entire population
    param_indices, neuron_indices = compute_layer_indices(layer_sizes)

    # Pre-allocate outputs
    outputs = np.zeros((pop_size, output_size), dtype=np.float64)
    updated_states = np.empty_like(population_states)  # No initialization - we overwrite everything
    updated_times = np.full(pop_size, current_time, dtype=np.float64)

    # Process each individual
    for i in prange(pop_size):
        individual_params = population[i]
        individual_prev_states = population_states[i]
        individual_prev_time = population_prev_times[i]

        # Predict with memory using pre-computed indices
        output, new_states, _ = predict_individual(
            individual_params, layer_sizes, activations, inputs,
            individual_prev_states, individual_prev_time, param_indices, neuron_indices
        )

        outputs[i] = output
        updated_states[i] = new_states

    return outputs, updated_states, updated_times