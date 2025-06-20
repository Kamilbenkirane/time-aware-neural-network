import numpy as np
from numba import njit

from ..network.utils import get_total_parameters
from ..network.activation import apply_activation
from ..network.neuron import discrete_time_leaky_integration
from ..network.utils import extract_layer_parameters

LOC = 0.0
SCALE = 1.0

@njit(fastmath=True, cache=False)
def initialize_individual(layer_sizes, seed=None, init_method="he"):
    """Initialize network for neural network with optional seed for reproducibility.

    Args:
        layer_sizes: List of layer sizes
        seed: Random seed for reproducibility
        init_method: Initialization method - "he", "xavier", or "normal"
    """
    total_parameters = get_total_parameters(layer_sizes)
    parameters = np.zeros(total_parameters, dtype=np.float64)

    # Handle seeding - must be done inside @njit function for Numba
    if seed is not None:
        np.random.seed(seed)

    # Initialize parameters layer by layer with appropriate scaling
    param_idx = 0
    num_layers = len(layer_sizes) - 1

    for layer in range(num_layers):
        fan_in = layer_sizes[layer]
        fan_out = layer_sizes[layer + 1]

        # Calculate appropriate scale based on initialization method
        if init_method == "he":
            # He initialization for ReLU activations
            weight_scale = np.sqrt(2.0 / fan_in)
        elif init_method == "xavier":
            # Xavier/Glorot initialization for tanh/sigmoid
            weight_scale = np.sqrt(2.0 / (fan_in + fan_out))
        else:
            # Default normal initialization
            weight_scale = SCALE

        # Initialize weights
        num_weights = fan_in * fan_out
        for i in range(num_weights):
            parameters[param_idx] = np.random.normal(LOC, weight_scale)
            param_idx += 1

        # Initialize biases (small positive values for ReLU to avoid dead neurons)
        for i in range(fan_out):
            if init_method == "he" and layer < num_layers - 1:  # For hidden layers with ReLU
                parameters[param_idx] = 0.01  # Small positive bias
            else:
                parameters[param_idx] = 0.0  # Zero for output layer
            param_idx += 1

        # Initialize alphas (time decay parameters) - should be in [0, 1]
        for i in range(fan_out):
            # Initialize closer to 0 for more responsive network (0.0 to 0.5)
            # High alpha = more memory but less responsive to inputs
            # Low alpha = less memory but more responsive to inputs
            parameters[param_idx] = 0.5 * np.random.random()
            param_idx += 1

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


@njit(fastmath=True, cache=True)
def individual_actions(individual, layer_sizes, layer_activations,
                    previous_states, previous_time, param_indices,
                    neuron_indices, timestamps, feature_values):
  """
  Compute actions for an individual across multiple timestamps.
  """
  n_timestamps = len(timestamps)
  actions = np.zeros(n_timestamps, dtype=np.int64)

  # Copy initial states to avoid modifying input
  current_states = np.copy(previous_states)
  current_time = previous_time

  for i in range(n_timestamps):
      # Create input tuple
      inputs = (timestamps[i], feature_values[i])

      # Get prediction - now returns only 2 values (removed current_time)
      outputs, current_states = predict_individual(
          individual, layer_sizes, layer_activations, inputs,
          current_states, current_time, param_indices, neuron_indices
      )

      # Update time for next iteration
      current_time = timestamps[i]

      # Compute action using argmax
      actions[i] = np.argmax(outputs)

  return actions