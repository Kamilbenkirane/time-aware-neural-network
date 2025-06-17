"""
Isolated Numba Genetic Algorithm - Self-Contained Implementation
Zero external imports except numpy, numba, math
"""

import numpy as np
import math
from numba import njit, prange

# Time resolution for discrete-time leaky integration
# NOTE: All time values throughout this module are in milliseconds
RESOLUTION_MS = 1  # 1 millisecond resolution


# ============================================================================
# BASIC WEIGHT MANAGEMENT FUNCTIONS
# ============================================================================

@njit(fastmath=True, cache=True)
def get_layer_parameters(layer_sizes, layer_id):
    """Calculate parameters for connection: layer[layer_id] -> layer[layer_id + 1]."""
    from_size = layer_sizes[layer_id]
    to_size = layer_sizes[layer_id + 1]
    # Weights: from_size * to_size + Biases: to_size + Alphas: to_size (one per neuron)
    return from_size * to_size + to_size + to_size


@njit(fastmath=True, cache=True)
def get_total_parameters(layer_sizes):
    """Calculate total parameters for time-aware neural network."""
    total = 0
    for layer_id in range(len(layer_sizes) - 1):
        total += get_layer_parameters(layer_sizes, layer_id)
    return total


# ============================================================================
# INDIVIDUAL PARAMETER INITIALIZATION (TIME-AWARE NETWORKS)
# ============================================================================



@njit(fastmath=True, cache=True)
def initialize_parameters(layer_sizes, seed=None):
    """Initialize parameters for neural network with optional seed for reproducibility."""
    total_parameters = get_total_parameters(layer_sizes)
    parameters = np.zeros(total_parameters, dtype=np.float64)
    
    # Handle seeding - must be done inside @njit function for Numba
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize with normal distribution (mean=0, std=0.3)
    for i in range(total_parameters):
        parameters[i] = np.random.normal(0.0, 0.3)
    
    return parameters


# ============================================================================
# POPULATION INITIALIZATION
# ============================================================================

@njit(fastmath=True, cache=True)
def initialize_population(pop_size, layer_sizes, seed=None):
    """Initialize a population of neural networks with random parameters."""
    total_parameters = get_total_parameters(layer_sizes)
    population = np.zeros((pop_size, total_parameters), dtype=np.float64)
    
    # Handle seeding
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize each individual
    for i in range(pop_size):
        # Initialize with normal distribution (mean=0, std=0.3)
        for j in range(total_parameters):
            population[i, j] = np.random.normal(0.0, 0.3)

    return population


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

@njit(fastmath=True, cache=True)
def apply_activation(value, activation_type):
    """Apply activation function based on integer type.
    
    Activation types:
    0 = Linear/Identity (no activation)
    1 = ReLU
    2 = Sigmoid  
    3 = Tanh
    4 = Leaky ReLU (alpha=0.01)
    """
    if activation_type == 0:    # Linear
        return value
    elif activation_type == 1:  # ReLU
        return max(0.0, value)
    elif activation_type == 2:  # Sigmoid
        # Clamp input to prevent overflow
        clamped = max(-500.0, min(500.0, value))
        return 1.0 / (1.0 + math.exp(-clamped))
    elif activation_type == 3:  # Tanh
        return math.tanh(value)
    elif activation_type == 4:  # Leaky ReLU
        return max(0.01 * value, value)
    else:
        return value  # Default to linear for unknown types


@njit(fastmath=True, cache=True)
def get_activation_name(activation_type):
    """Get string name for activation type (for testing/debugging)."""
    if activation_type == 0:
        return "linear"
    elif activation_type == 1:
        return "relu"
    elif activation_type == 2:
        return "sigmoid"
    elif activation_type == 3:
        return "tanh"
    elif activation_type == 4:
        return "leaky_relu"
    else:
        return "unknown"


# ============================================================================
# NEURAL NETWORK PREDICTION
# ============================================================================

@njit(fastmath=True, cache=True)
def discrete_time_leaky_integration(prev_state, linear_input, time_diff, alpha):
    """Apply discrete-time leaky integration with time-aware decay."""
    # Clamp alpha to valid range [0, 1]
    alpha_clamped = max(0.0, min(1.0, alpha))
    
    num_steps = max(0, int(time_diff / RESOLUTION_MS))
    decay_factor = alpha_clamped ** num_steps
    decayed_state = prev_state * decay_factor
    return alpha_clamped * decayed_state + (1.0 - alpha_clamped) * linear_input


@njit(fastmath=True, cache=True)
def compute_layer_indices(layer_sizes):
    """Pre-compute parameter start indices and neuron mappings for efficient access."""
    num_layers = len(layer_sizes) - 1
    param_indices = np.zeros(num_layers + 1, dtype=np.int64)
    neuron_indices = np.zeros(num_layers + 1, dtype=np.int64)
    
    # Parameter indices (weights + biases + alphas)
    for i in range(num_layers):
        param_indices[i + 1] = param_indices[i] + get_layer_parameters(layer_sizes, i)
    
    # Neuron state indices (hidden + output layers only)
    for i in range(num_layers):
        neuron_indices[i + 1] = neuron_indices[i] + layer_sizes[i + 1]
    
    return param_indices, neuron_indices


@njit(fastmath=True, cache=True)
def predict_individual(parameters, layer_sizes, activations, inputs, 
                      prev_states, prev_time, param_indices, neuron_indices):
    """
    Predict output for a single neural network with temporal memory.
    
    Args:
        parameters: Flat array of network parameters (weights + biases + alphas)
        layer_sizes: Array of layer sizes [input, hidden1, ..., output]
        activations: Array of activation types for each layer transition
        inputs: Tuple (current_time, x_vector)
        prev_states: Previous neuron states (pre-activation)
        prev_time: Previous evaluation timestamp
        param_indices: Pre-computed parameter indices for each layer
        neuron_indices: Pre-computed neuron state indices for each layer
    
    Returns:
        (outputs, new_states, current_time): Updated outputs and states
    """
    # Input validation
    if len(inputs) != 2:
        return np.zeros(layer_sizes[-1], dtype=np.float64), prev_states, 0.0
    
    current_time, x_vector = inputs
    
    # Validate input dimensions
    if len(x_vector) != layer_sizes[0]:
        return np.zeros(layer_sizes[-1], dtype=np.float64), prev_states, current_time
    
    if len(prev_states) != sum(layer_sizes[1:]):
        return np.zeros(layer_sizes[-1], dtype=np.float64), prev_states, current_time
        
    time_diff = max(0.0, current_time - prev_time)  # Ensure non-negative time diff
    
    # Initialize with input layer
    current_values = x_vector
    # Pre-allocate new states array (no initialization needed - we overwrite everything)
    new_states = np.empty_like(prev_states)
    
    # Forward pass through each layer
    for layer_id in range(len(layer_sizes) - 1):
        from_size = layer_sizes[layer_id]
        to_size = layer_sizes[layer_id + 1]
        
        # Extract parameters using pre-computed indices
        start_idx = param_indices[layer_id]
        weights_size = from_size * to_size
        weights_end = start_idx + weights_size
        biases_end = weights_end + to_size
        alphas_end = biases_end + to_size
        
        weights_flat = parameters[start_idx:weights_end]
        biases = parameters[weights_end:biases_end]
        alphas = parameters[biases_end:alphas_end]
        
        # Reshape weights for efficient BLAS operation
        weights_matrix = weights_flat.reshape((from_size, to_size))
        
        # Compute linear transformation using optimized BLAS
        linear_outputs = np.dot(current_values, weights_matrix) + biases
        
        # Apply time-aware memory integration for each neuron
        neuron_start = neuron_indices[layer_id]
        next_values = np.zeros(to_size, dtype=np.float64)
        
        for i in range(to_size):
            neuron_idx = neuron_start + i
            prev_neuron_state = prev_states[neuron_idx]
            
            # Discrete-time leaky integration (pre-activation)
            integrated_state = discrete_time_leaky_integration(
                prev_neuron_state, linear_outputs[i], time_diff, alphas[i]
            )
            
            # Apply activation function
            next_values[i] = apply_activation(integrated_state, activations[layer_id])
            
            # Store pre-activation state for next iteration
            new_states[neuron_idx] = integrated_state
        
        current_values = next_values
    
    return current_values, new_states, current_time


@njit(fastmath=True, cache=True)
def predict_population(population, layer_sizes, activations, inputs,
                      population_states, population_prev_times):
    """
    Predict outputs for entire population with temporal memory.
    
    Args:
        population: 2D array (pop_size, total_parameters)
        layer_sizes: Array of layer sizes
        activations: Array of activation types (shared by all individuals)
        inputs: Tuple (current_time, x_vector)
        population_states: 2D array (pop_size, total_neurons) - previous states
        population_prev_times: 1D array (pop_size,) - previous timestamps
        
    Returns:
        (outputs, updated_states, updated_times): Population results
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
    for i in range(pop_size):
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


@njit(fastmath=True, cache=True)
def reset_population_memory(layer_sizes, pop_size):
    """Create fresh memory states for a population."""
    total_neurons = sum(layer_sizes[1:])  # Exclude input layer
    population_states = np.zeros((pop_size, total_neurons), dtype=np.float64)
    population_times = np.zeros(pop_size, dtype=np.float64)
    return population_states, population_times


@njit(fastmath=True, cache=True)
def predict_individual_stateless(parameters, layer_sizes, activations, x_vector):
    """
    Simple stateless prediction (for compatibility/testing).
    Equivalent to calling predict_individual with zero previous states.
    """
    current_time = 0.0
    inputs = (current_time, x_vector)
    total_neurons = sum(layer_sizes[1:])
    prev_states = np.zeros(total_neurons, dtype=np.float64)
    prev_time = 0.0
    
    # Pre-compute indices for this call
    param_indices, neuron_indices = compute_layer_indices(layer_sizes)
    
    outputs, _, _ = predict_individual(
        parameters, layer_sizes, activations, inputs, prev_states, prev_time,
        param_indices, neuron_indices
    )
    return outputs


# ============================================================================
# SELECTION OPERATIONS FOR GENETIC ALGORITHMS
# ============================================================================

@njit(fastmath=True, cache=True)
def tournament_selection(population, fitness_scores, tournament_size, num_parents, seed=None):
    """
    Tournament selection for genetic algorithms.
    
    Runs tournaments of k randomly selected individuals, chooses winner with best fitness.
    Binary tournament (k=2) is most common. Larger k = higher selection pressure.
    
    Args:
        population: 2D array (pop_size, num_parameters) - population of individuals
        fitness_scores: 1D array (pop_size,) - fitness for each individual (higher = better)
        tournament_size: int - individuals per tournament (2-7 typical, 2 most common)
        num_parents: int - number of parents to select
        seed: optional int - for reproducible results
        
    Returns:
        selected_parents: 2D array (num_parents, num_parameters) - selected individuals
        selected_indices: 1D array (num_parents,) - indices in original population
    """
    if seed is not None:
        np.random.seed(seed)
    
    pop_size = population.shape[0]
    num_parameters = population.shape[1]
    
    # Clamp parameters to valid ranges (input validation)
    tournament_size = max(1, min(tournament_size, pop_size))
    num_parents = max(1, num_parents)
    
    # Pre-allocate result arrays (avoid allocation in loops)
    selected_parents = np.zeros((num_parents, num_parameters), dtype=np.float64)
    selected_indices = np.zeros(num_parents, dtype=np.int64)
    
    # Pre-allocate tournament array for reuse (cache-friendly)
    tournament_indices = np.zeros(tournament_size, dtype=np.int64)
    
    # Run tournaments (nested loops over vectorized for small arrays)
    for parent_idx in range(num_parents):
        # Select random tournament participants
        for i in range(tournament_size):
            tournament_indices[i] = np.random.randint(0, pop_size)
        
        # Find tournament winner (best fitness)
        best_idx = 0
        best_fitness = fitness_scores[tournament_indices[0]]
        
        for i in range(1, tournament_size):
            candidate_fitness = fitness_scores[tournament_indices[i]]
            if candidate_fitness > best_fitness:
                best_fitness = candidate_fitness
                best_idx = i
        
        # Store winner
        winner_idx = tournament_indices[best_idx]
        selected_indices[parent_idx] = winner_idx
        selected_parents[parent_idx] = population[winner_idx]
    
    return selected_parents, selected_indices
