import math

import numpy as np
from numba import prange, njit

RESOLUTION = .001  # 1 millisecond resolution

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

@njit(fastmath=True, cache=True)
def individual_actions(individual, layer_sizes, layer_activations, previous_states, previous_time, param_indices, neuron_indices, timestamps, feature_values):
    n_timestamps = len(timestamps)
    actions = np.zeros(n_timestamps, dtype=np.int64)
    for i in range(n_timestamps):
        current_time = timestamps[i]
        values = feature_values[i]
        inputs = (current_time, values)
        current_values, new_states = predict_individual(individual, layer_sizes, layer_activations, inputs, previous_states, previous_time, param_indices, neuron_indices)
        previous_states = new_states
        previous_time = current_time
        actions[i] = np.argmax(current_values)

    return actions

@njit(fastmath=True, cache=False)
def predict_individual(parameters, layer_sizes, activations, inputs,
                       prev_states, prev_time, param_indices, neuron_indices):
    """
    Predict output for a single neural network with temporal memory.

    Args:
        parameters: Flat array of network network (weights + biases + alphas)
        layer_sizes: Array of layer sizes [input, hidden1, ..., output]
        activations: Array of activation types for each layer transition
        inputs: Tuple (current_time, x_vector)
        prev_states: Previous neuron states (pre-activation)
        prev_time: Previous evaluation timestamp
        param_indices: Pre-computed parameter indices for each layer
        neuron_indices: Pre-computed neuron state indices for each layer

    Returns:
        (outputs, new_states): Updated outputs and states
    """
    # Input validation
    if len(inputs) != 2:
        return np.zeros(layer_sizes[-1], dtype=np.float64), prev_states

    current_time, x_vector = inputs

    # Validate input dimensions
    if len(x_vector) != layer_sizes[0]:
        return np.zeros(layer_sizes[-1], dtype=np.float64), prev_states

    if len(prev_states) != sum(layer_sizes[1:]):
        return np.zeros(layer_sizes[-1], dtype=np.float64), prev_states

    time_diff = max(0.0, current_time - prev_time)  # Ensure non-negative time diff

    # Initialize with input layer
    current_values = x_vector
    # Pre-allocate new states array (no initialization needed - we overwrite everything)
    new_states = np.empty_like(prev_states)

    # Forward pass through each layer
    for layer_id in range(len(layer_sizes) - 1):
        from_size = layer_sizes[layer_id]
        to_size = layer_sizes[layer_id + 1]

        # Extract network using reusable function
        weights_flat, biases, alphas, weights_matrix = extract_layer_parameters(
            parameters, param_indices, layer_id, from_size, to_size
        )

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

    return current_values, new_states


@njit(fastmath=True, cache=False)
def extract_layer_parameters(parameters, param_indices, layer_id, from_size, to_size):
    """
    Extract weights, biases, and alphas for a specific layer.
    """
    # Calculate parameter slice indices
    start_idx = param_indices[layer_id]
    weights_size = from_size * to_size
    weights_end = start_idx + weights_size
    biases_end = weights_end + to_size
    alphas_end = biases_end + to_size

    # Extract parameter arrays
    weights_flat = parameters[start_idx:weights_end]
    biases = parameters[weights_end:biases_end]
    alphas = parameters[biases_end:alphas_end]

    # Reshape weights for matrix operations
    weights_matrix = weights_flat.reshape((from_size, to_size))

    return weights_flat, biases, alphas, weights_matrix

@njit(fastmath=True, cache=False)
def discrete_time_leaky_integration(prev_state, linear_input, time_diff, alpha):
    """Apply discrete-time leaky integration with time-aware decay."""
    # Clamp alpha to valid range [0, 1]
    alpha_clamped = max(0.0, min(1.0, alpha))

    num_steps = max(0, int(time_diff / RESOLUTION))
    decay_factor = alpha_clamped ** num_steps
    decayed_state = prev_state * decay_factor
    return alpha_clamped * decayed_state + (1.0 - alpha_clamped) * linear_input

@njit(fastmath=True, cache=False)
def apply_activation(value, activation_type):
    """Apply activation function based on integer type.

    Activation types:
    0 = Linear/Identity (no activation)
    1 = ReLU
    2 = Sigmoid
    3 = Tanh
    4 = Leaky ReLU (alpha=0.01)
    """
    if activation_type == 0:  # Linear
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