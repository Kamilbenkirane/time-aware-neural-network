import numpy as np
from numba import njit

from ..network.utils import get_total_parameters
from ..network.activation import apply_activation
from ..network.neuron import discrete_time_leaky_integration
from ..network.utils import extract_layer_parameters

LOC = 0.0
SCALE = 1.0

@njit(fastmath=True, cache=False)
def initialize_individual(layer_sizes, seed=None):
    """Initialize network for neural network with optional seed for reproducibility."""
    total_parameters = get_total_parameters(layer_sizes)
    parameters = np.zeros(total_parameters, dtype=np.float64)

    # Handle seeding - must be done inside @njit function for Numba
    if seed is not None:
        np.random.seed(seed)

    for i in range(total_parameters):
        parameters[i] = np.random.normal(LOC, SCALE)

    return parameters


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

    return current_values, new_states, current_time