import math

from numba import njit


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
        clamped = max(-50.0, min(50.0, value))
        return 1.0 / (1.0 + math.exp(-clamped))
    elif activation_type == 3:  # Tanh
        return math.tanh(value)
    elif activation_type == 4:  # Leaky ReLU
        return max(0.01 * value, value)
    else:
        return value  # Default to linear for unknown types


@njit(fastmath=True, cache=False)
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