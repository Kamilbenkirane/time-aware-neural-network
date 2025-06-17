"""
Pure Function-Based Individual Operations with Numba JIT Compilation.

This module implements individual operations using ONLY pure @njit functions
with raw numpy arrays. No classes, no object instantiation, zero Python overhead.

Architecture:
Raw Weight Array → @njit Functions → Results

Key principles:
- Zero classes in computation
- Raw array operations only  
- Stateless pure functions
- Immutable operations (return new arrays)
- Maximum @njit optimization
"""

import numpy as np
import math
from numba import njit
from typing import Tuple


# ============================================================================
# WEIGHT MANAGEMENT FUNCTIONS
# ============================================================================

@njit(fastmath=True, cache=True)
def get_total_weights_count(input_size, hidden_size, output_size):
    """
    Calculate total number of weights needed for neural network.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden neurons
        output_size: Number of output neurons
    
    Returns:
        Total number of weights (int)
    """
    return input_size * hidden_size + hidden_size + output_size * hidden_size + output_size


@njit(fastmath=True, cache=True)
def initialize_individual_weights(input_size, hidden_size, output_size, seed=None):
    """
    Initialize random weights for an individual.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden neurons
        output_size: Number of output neurons
        seed: Random seed (optional)
    
    Returns:
        1D numpy array of weights
    """
    if seed is not None:
        np.random.seed(seed)
    
    total_weights = get_total_weights_count(input_size, hidden_size, output_size)
    weights = np.random.randn(total_weights).astype(np.float64) * 0.3
    return weights


@njit(fastmath=True, cache=True)
def clone_individual_weights(weights):
    """
    Create a copy of individual weights.
    
    Args:
        weights: 1D weight array
    
    Returns:
        Copy of weight array
    """
    return weights.copy()


# ============================================================================
# NEURAL NETWORK STATE FUNCTIONS
# ============================================================================

@njit(fastmath=True, cache=True)
def create_individual_nn_state(hidden_size, output_size):
    """
    Create initial neural network temporal state.
    
    Args:
        hidden_size: Number of hidden neurons
        output_size: Number of output neurons
    
    Returns:
        Tuple of (hidden_prev, hidden_time, output_prev, output_time)
    """
    hidden_prev = np.zeros(hidden_size, dtype=np.float64)
    hidden_time = 0.0
    output_prev = np.zeros(output_size, dtype=np.float64)
    output_time = 0.0
    
    return hidden_prev, hidden_time, output_prev, output_time


@njit(fastmath=True, cache=True)
def reset_individual_nn_state(nn_state):
    """
    Reset neural network temporal state in-place.
    
    Args:
        nn_state: Neural network state tuple
    
    Returns:
        Reset state tuple
    """
    hidden_prev, hidden_time, output_prev, output_time = nn_state
    hidden_prev.fill(0.0)
    output_prev.fill(0.0)
    return hidden_prev, 0.0, output_prev, 0.0


# ============================================================================
# NEURAL NETWORK FORWARD PASS FUNCTIONS
# ============================================================================

@njit(fastmath=True, cache=True)
def extract_individual_weights(weight_array, input_size, hidden_size, output_size):
    """
    Extract weight matrices from flat weight array for individual.
    
    Args:
        weight_array: Flat 1D weight array
        input_size: Number of input features
        hidden_size: Number of hidden neurons
        output_size: Number of output neurons
    
    Returns:
        Tuple of (hidden_weights, hidden_bias, output_weights, output_bias)
    """
    # Calculate indices
    hw_end = hidden_size * input_size
    hb_end = hw_end + hidden_size
    ow_end = hb_end + output_size * hidden_size
    ob_end = ow_end + output_size
    
    # Extract and reshape
    hidden_weights = weight_array[:hw_end].reshape(hidden_size, input_size)
    hidden_bias = weight_array[hw_end:hb_end]
    output_weights = weight_array[hb_end:ow_end].reshape(output_size, hidden_size)
    output_bias = weight_array[ow_end:ob_end]
    
    return hidden_weights, hidden_bias, output_weights, output_bias


@njit(fastmath=True, cache=True)
def compute_temporal_decay(current_time, prev_time):
    """
    Compute exponential temporal decay factor.
    
    Args:
        current_time: Current timestamp
        prev_time: Previous timestamp
    
    Returns:
        Decay factor between 0 and 1
    """
    if prev_time <= 0.0:
        return 0.0
    
    time_diff = max(0.0, min(current_time - prev_time, 50.0))
    return math.exp(-time_diff)


@njit(fastmath=True, cache=True)
def individual_layer_forward_pass(x, weights, bias, prev_values, prev_time, current_time, alpha):
    """
    Forward pass through a single time-aware layer for individual.
    
    Args:
        x: Input array
        weights: Weight matrix
        bias: Bias array
        prev_values: Previous layer outputs (modified in-place)
        prev_time: Previous timestamp
        current_time: Current timestamp
        alpha: Memory strength parameter
    
    Returns:
        Tuple of (output, new_prev_time)
    """
    # Linear transformation
    linear_out = np.dot(weights, x) + bias
    
    # Temporal decay
    decay_factor = compute_temporal_decay(current_time, prev_time)
    
    # Add temporal memory
    memory_contribution = alpha * prev_values * decay_factor
    memory_out = linear_out + memory_contribution
    
    # Apply activation
    output = np.tanh(memory_out)
    
    # Update previous values in-place
    prev_values[:] = output
    
    return output, current_time


@njit(fastmath=True, cache=True)
def get_individual_action(weight_array, input_value, current_time, nn_state,
                         input_size, hidden_size, output_size, alpha):
    """
    Get action from individual using neural network forward pass.
    
    Args:
        weight_array: Flat weight array
        input_value: Single input value
        current_time: Current timestamp
        nn_state: Neural network state tuple
        input_size: Number of input features
        hidden_size: Number of hidden neurons
        output_size: Number of output neurons
        alpha: Memory strength parameter
    
    Returns:
        Tuple of (action_index, updated_nn_state)
    """
    hidden_prev, hidden_time, output_prev, output_time = nn_state
    
    # Extract weights
    hidden_weights, hidden_bias, output_weights, output_bias = extract_individual_weights(
        weight_array, input_size, hidden_size, output_size
    )
    
    # Prepare input
    x_input = np.array([input_value], dtype=np.float64)
    
    # Hidden layer forward pass
    hidden_out, new_hidden_time = individual_layer_forward_pass(
        x_input, hidden_weights, hidden_bias, hidden_prev, hidden_time, current_time, alpha
    )
    
    # Output layer forward pass
    output_out, new_output_time = individual_layer_forward_pass(
        hidden_out, output_weights, output_bias, output_prev, output_time, current_time, alpha
    )
    
    # Get action (argmax)
    action_index = np.argmax(output_out)
    
    # Return updated state
    updated_state = (hidden_prev, new_hidden_time, output_prev, new_output_time)
    
    return action_index, updated_state


# ============================================================================
# GENETIC OPERATION FUNCTIONS
# ============================================================================

@njit(fastmath=True, cache=True)
def mutate_individual_weights(weights, mutation_rate, mutation_strength):
    """
    Mutate individual weights using Gaussian mutation.
    
    Args:
        weights: 1D weight array
        mutation_rate: Probability of mutation per weight
        mutation_strength: Standard deviation for mutation
    
    Returns:
        Mutated weight array (new array)
    """
    # Pre-allocate result array to avoid race condition in parallel context
    mutated_weights = np.empty(len(weights), dtype=np.float64)
    
    # Copy weights element by element to avoid .copy() race condition
    for i in range(len(weights)):
        mutated_weights[i] = weights[i]
        if np.random.random() < mutation_rate:
            mutation_value = np.random.normal(0.0, mutation_strength)
            mutated_weights[i] += mutation_value
    
    return mutated_weights


@njit(fastmath=True, cache=True)
def crossover_individuals(parent1_weights, parent2_weights, crossover_rate):
    """
    Perform uniform crossover between two individuals.
    
    Args:
        parent1_weights: First parent weight array
        parent2_weights: Second parent weight array
        crossover_rate: Probability of crossover occurring
    
    Returns:
        Tuple of (child1_weights, child2_weights)
    """
    # Pre-allocate result arrays to avoid race condition in parallel context
    child1_weights = np.empty(len(parent1_weights), dtype=np.float64)
    child2_weights = np.empty(len(parent2_weights), dtype=np.float64)
    
    if np.random.random() >= crossover_rate:
        # No crossover - copy parents element by element
        for i in range(len(parent1_weights)):
            child1_weights[i] = parent1_weights[i]
            child2_weights[i] = parent2_weights[i]
    else:
        # Uniform crossover
        for i in range(len(parent1_weights)):
            if np.random.random() < 0.5:
                child1_weights[i] = parent1_weights[i]
                child2_weights[i] = parent2_weights[i]
            else:
                child1_weights[i] = parent2_weights[i]
                child2_weights[i] = parent1_weights[i]
    
    return child1_weights, child2_weights


# ============================================================================
# CONVENIENCE WRAPPER FUNCTIONS (MINIMAL - FOR API COMPATIBILITY)
# ============================================================================

class IndividualWrapper:
    """
    Minimal wrapper for API compatibility with fitness functions.
    Only used at the interface level - all computation is pure functions.
    """
    
    def __init__(self, weights, input_size=1, hidden_size=10, output_size=3, alpha=1.0):
        self.weights = weights
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha
        self.nn_state = create_individual_nn_state(hidden_size, output_size)
    
    def get_weights(self):
        """Get weights for fitness function."""
        return self.weights
    
    def get_action(self, input_value, current_time):
        """Get action (for compatibility)."""
        action, self.nn_state = get_individual_action(
            self.weights, input_value, current_time, self.nn_state,
            self.input_size, self.hidden_size, self.output_size, self.alpha
        )
        return action
    
    def reset_state(self):
        """Reset neural network state."""
        self.nn_state = reset_individual_nn_state(self.nn_state)
    
    @property
    def total_weights(self):
        """Total number of weights."""
        return len(self.weights)


def create_individual_wrapper(weights, input_size=1, hidden_size=10, output_size=3, alpha=1.0):
    """
    Create minimal wrapper for API compatibility.
    
    Args:
        weights: Weight array
        input_size: Neural network input size
        hidden_size: Neural network hidden size
        output_size: Neural network output size
        alpha: Temporal memory strength
    
    Returns:
        IndividualWrapper instance
    """
    return IndividualWrapper(weights, input_size, hidden_size, output_size, alpha)


# ============================================================================
# COMPATIBILITY ALIASES
# ============================================================================

# For drop-in replacement
Individual = IndividualWrapper


# ============================================================================
# PERFORMANCE TESTING
# ============================================================================

def benchmark_pure_individual_performance():
    """Benchmark pure individual function performance."""
    import time
    
    # Test parameters
    input_size, hidden_size, output_size = 1, 10, 3
    alpha = 1.0
    num_iterations = 10000
    
    # Initialize individual
    weights = initialize_individual_weights(input_size, hidden_size, output_size, seed=42)
    nn_state = create_individual_nn_state(hidden_size, output_size)
    
    # Generate test data
    np.random.seed(42)
    inputs = np.random.randn(num_iterations)
    timestamps = np.arange(num_iterations, dtype=np.float64)
    
    # Warm up compilation
    for i in range(10):
        action, nn_state = get_individual_action(
            weights, inputs[i], timestamps[i], nn_state,
            input_size, hidden_size, output_size, alpha
        )
    
    # Reset and benchmark
    nn_state = reset_individual_nn_state(nn_state)
    start_time = time.time()
    
    for i in range(num_iterations):
        action, nn_state = get_individual_action(
            weights, inputs[i], timestamps[i], nn_state,
            input_size, hidden_size, output_size, alpha
        )
    
    total_time = time.time() - start_time
    
    print(f"Pure Individual Function Performance:")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Time per action: {total_time/num_iterations*1000:.4f} ms")
    print(f"Actions per second: {num_iterations/total_time:.0f}")
    
    return {
        'total_time': total_time,
        'time_per_action': total_time / num_iterations,
        'actions_per_second': num_iterations / total_time
    }


if __name__ == "__main__":
    print("Pure Function-Based Individual Operations")
    print("=" * 50)
    print("✅ Zero classes in core computation")
    print("⚡ Pure @njit functions with raw arrays")
    print("🚀 Maximum performance optimization")
    
    # Test basic operations
    weights = initialize_individual_weights(1, 5, 3, seed=42)
    print(f"\nInitialized weights: {len(weights)} values")
    
    # Test neural network operations
    nn_state = create_individual_nn_state(5, 3)
    action, nn_state = get_individual_action(weights, 0.5, 1.0, nn_state, 1, 5, 3, 1.0)
    print(f"Action for input 0.5: {action}")
    
    # Test genetic operations
    mutated = mutate_individual_weights(weights, 0.1, 0.1)
    print(f"Mutation changed {np.sum(weights != mutated)} weights")
    
    # Performance benchmark
    print("\nRunning performance benchmark...")
    results = benchmark_pure_individual_performance()
    
    print(f"\n🎯 Pure function-based individual operations complete!")
    print(f"⚡ {results['actions_per_second']:,.0f} actions per second")
    print(f"🚀 Ready for pure function-based genetic algorithm!")