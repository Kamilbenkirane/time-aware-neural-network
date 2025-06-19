import numpy as np
from numba import njit

from ..network import get_layer_parameters

RESOLUTION = .001  # 1 millisecond resolution

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