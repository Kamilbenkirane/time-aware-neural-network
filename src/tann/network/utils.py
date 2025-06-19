from numba import njit
import numpy as np

@njit(fastmath=True, cache=False)
def get_layer_parameters(layer_sizes, layer_id):
    """Calculate network for connection: layer[layer_id] -> layer[layer_id + 1]."""
    from_size = layer_sizes[layer_id]
    to_size = layer_sizes[layer_id + 1]
    # Weights: from_size * to_size + Biases: to_size + Alphas: to_size (one per neuron)
    return from_size * to_size + to_size + to_size


@njit(fastmath=True, cache=False)
def get_total_parameters(layer_sizes):
    """Calculate total network for time-aware neural network."""
    total = 0
    for layer_id in range(len(layer_sizes) - 1):
        total += get_layer_parameters(layer_sizes, layer_id)
    return total


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
