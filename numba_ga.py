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
    """Calculate network for connection: layer[layer_id] -> layer[layer_id + 1]."""
    from_size = layer_sizes[layer_id]
    to_size = layer_sizes[layer_id + 1]
    # Weights: from_size * to_size + Biases: to_size + Alphas: to_size (one per neuron)
    return from_size * to_size + to_size + to_size


@njit(fastmath=True, cache=True)
def get_total_parameters(layer_sizes):
    """Calculate total network for time-aware neural network."""
    total = 0
    for layer_id in range(len(layer_sizes) - 1):
        total += get_layer_parameters(layer_sizes, layer_id)
    return total


@njit(fastmath=True, cache=True)
def extract_layer_parameters(parameters, param_indices, layer_id, from_size, to_size):
    """
    Extract weights, biases, and alphas for a specific layer.
    
    Args:
        parameters: Flat parameter array for entire network
        param_indices: Pre-computed parameter start indices for each layer
        layer_id: Which layer to extract (0-indexed)
        from_size: Input size for this layer
        to_size: Output size for this layer
        
    Returns:
        weights_flat: Flat weight array (from_size * to_size)
        biases: Bias array (to_size)
        alphas: Alpha array (to_size) for time-aware integration
        weights_matrix: Reshaped weight matrix (from_size, to_size)
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


# ============================================================================
# INDIVIDUAL PARAMETER INITIALIZATION (TIME-AWARE NETWORKS)
# ============================================================================



@njit(fastmath=True, cache=True)
def initialize_individual(layer_sizes, seed=None):
    """Initialize network for neural network with optional seed for reproducibility."""
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
    """Initialize a population of neural networks with random network."""
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


@njit(fastmath=True, cache=True)
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
    
    # Clamp network to valid ranges (input validation)
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


# ============================================================================
# CROSSOVER OPERATIONS FOR GENETIC ALGORITHMS
# ============================================================================

@njit(fastmath=True, cache=True)
def align_neurons_correlation(parent1, parent2, layer_sizes, seed=None):
    """
    Align neurons between two parents using weight correlation analysis.
    
    Solves the permutation problem by matching neurons with similar weight patterns.
    For each layer, computes correlations between neuron weight vectors and rearranges 
    parent2's neurons to best match parent1's neuron ordering.
    
    Args:
        parent1: Flat parameter array for first parent
        parent2: Flat parameter array for second parent  
        layer_sizes: Array of layer sizes [input, hidden1, ..., output]
        seed: Optional seed for reproducible results
        
    Returns:
        aligned_parent1: Original parent1 (unchanged)
        aligned_parent2: Parent2 with neurons rearranged to align with parent1
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Work with copies to avoid modifying originals
    aligned_parent1 = parent1.copy()
    aligned_parent2 = parent2.copy()
    
    # Pre-compute parameter indices for efficient access
    param_indices, _ = compute_layer_indices(layer_sizes)
    num_layers = len(layer_sizes) - 1
    
    # Process each layer sequentially (excluding input layer)
    for layer_id in range(num_layers):
        from_size = layer_sizes[layer_id]
        to_size = layer_sizes[layer_id + 1]
        
        if to_size <= 1:
            continue  # Skip layers with single neuron (nothing to align)
        
        # Extract network for both parents using reusable function
        w1_flat, b1, a1, w1 = extract_layer_parameters(
            aligned_parent1, param_indices, layer_id, from_size, to_size
        )
        w2_flat, b2, a2, w2 = extract_layer_parameters(
            aligned_parent2, param_indices, layer_id, from_size, to_size
        )
        
        # Calculate indices needed for parameter storage later
        start_idx = param_indices[layer_id]
        weights_size = from_size * to_size
        weights_end = start_idx + weights_size
        biases_end = weights_end + to_size
        
        # Compute correlation matrix between all neuron pairs
        corr_matrix = np.zeros((to_size, to_size), dtype=np.float64)
        
        for i in range(to_size):  # neurons in parent1
            for j in range(to_size):  # neurons in parent2
                # Get incoming weight vectors + bias for correlation
                weights1 = w1[:, i]  # incoming to neuron i in parent1
                weights2 = w2[:, j]  # incoming to neuron j in parent2
                
                # Create extended vectors including bias
                n = from_size
                vec1 = np.zeros(n + 1, dtype=np.float64)
                vec2 = np.zeros(n + 1, dtype=np.float64)
                vec1[:n] = weights1
                vec1[n] = b1[i]
                vec2[:n] = weights2
                vec2[n] = b2[j]
                
                # Compute Pearson correlation coefficient
                mean1 = np.mean(vec1)
                mean2 = np.mean(vec2)
                
                numerator = 0.0
                sum_sq1 = 0.0
                sum_sq2 = 0.0
                
                for k in range(n + 1):
                    d1 = vec1[k] - mean1
                    d2 = vec2[k] - mean2
                    numerator += d1 * d2
                    sum_sq1 += d1 * d1
                    sum_sq2 += d2 * d2
                
                denominator = math.sqrt(sum_sq1 * sum_sq2)
                if denominator < 1e-10:
                    corr_matrix[i, j] = 0.0
                else:
                    corr_matrix[i, j] = numerator / denominator
        
        # Find optimal neuron matching using greedy algorithm
        matching = np.zeros(to_size, dtype=np.int64)
        used = np.zeros(to_size, dtype=np.bool_)
        
        for i in range(to_size):
            best_j = -1
            best_corr = -2.0  # Below minimum possible correlation
            
            for j in range(to_size):
                if not used[j] and corr_matrix[i, j] > best_corr:
                    best_corr = corr_matrix[i, j]
                    best_j = j
            
            if best_j >= 0:
                matching[i] = best_j
                used[best_j] = True
            else:
                # Fallback: find first unused position
                for j in range(to_size):
                    if not used[j]:
                        matching[i] = j
                        used[j] = True
                        break
                else:
                    matching[i] = i  # Should not happen in well-formed input
        
        # Rearrange parent2's network according to matching
        # Create rearranged weight matrix, biases, and alphas
        w2_aligned = np.zeros((from_size, to_size), dtype=np.float64)
        b2_aligned = np.zeros(to_size, dtype=np.float64)
        a2_aligned = np.zeros(to_size, dtype=np.float64)
        
        for i in range(to_size):
            j = matching[i]  # neuron j in parent2 matches position i
            w2_aligned[:, i] = w2[:, j]  # incoming weights
            b2_aligned[i] = b2[j]        # bias
            a2_aligned[i] = a2[j]        # alpha
        
        # Store rearranged network back into aligned_parent2
        w2_aligned_flat = w2_aligned.flatten()
        aligned_parent2[start_idx:weights_end] = w2_aligned_flat
        aligned_parent2[weights_end:biases_end] = b2_aligned
        aligned_parent2[biases_end:biases_end + to_size] = a2_aligned
        
        # Rearrange outgoing weights to next layer (if not output layer)
        if layer_id < num_layers - 1:
            next_layer_id = layer_id + 1
            next_from_size = layer_sizes[next_layer_id]
            next_to_size = layer_sizes[next_layer_id + 1]
            
            next_start_idx = param_indices[next_layer_id]
            next_weights_size = next_from_size * next_to_size
            next_weights_end = next_start_idx + next_weights_size
            
            # Extract next layer weights  
            next_w_flat = aligned_parent2[next_start_idx:next_weights_end]
            next_w = next_w_flat.reshape((next_from_size, next_to_size))
            
            # Rearrange rows according to current layer matching
            next_w_aligned = np.zeros_like(next_w)
            for i in range(to_size):
                j = matching[i]
                next_w_aligned[i, :] = next_w[j, :]
            
            # Store rearranged next layer weights
            next_w_aligned_flat = next_w_aligned.flatten()
            aligned_parent2[next_start_idx:next_weights_end] = next_w_aligned_flat
    
    return aligned_parent1, aligned_parent2


@njit(fastmath=True, cache=True)
def safe_arithmetic_crossover(parent1, parent2, layer_sizes, alpha=0.5, seed=None):
    """
    Perform safe arithmetic crossover after neuron alignment.
    
    First aligns neurons between parents using correlation analysis to solve 
    the permutation problem, then performs arithmetic blending of aligned network.
    This is the research-proven method for effective neural network crossover.
    
    Args:
        parent1: Flat parameter array for first parent
        parent2: Flat parameter array for second parent
        layer_sizes: Array of layer sizes [input, hidden1, ..., output]
        alpha: Blending factor in [0,1]. 0.0=parent2, 1.0=parent1, 0.5=average
        seed: Optional seed for reproducible alignment
        
    Returns:
        offspring: Blended parameter array combining both aligned parents
    """
    # Input validation
    alpha_clamped = max(0.0, min(1.0, alpha))  # Clamp to valid range [0,1]
    
    # Validate parent compatibility
    if len(parent1) != len(parent2):
        # Return copy of first parent if incompatible shapes
        return parent1.copy()
    
    # Align neurons first to solve permutation problem
    aligned_parent1, aligned_parent2 = align_neurons_correlation(
        parent1, parent2, layer_sizes, seed
    )
    
    # Handle edge cases for alpha (AFTER alignment)
    if alpha_clamped <= 0.0:
        return aligned_parent2.copy()  # Return aligned second parent
    elif alpha_clamped >= 1.0:
        return aligned_parent1.copy()  # Return aligned first parent
    
    # Perform arithmetic crossover: offspring = α * p1 + (1-α) * p2
    total_params = len(aligned_parent1)
    offspring = np.zeros(total_params, dtype=np.float64)
    
    for i in range(total_params):
        offspring[i] = alpha_clamped * aligned_parent1[i] + (1.0 - alpha_clamped) * aligned_parent2[i]
    
    return offspring


# =============================================================================
# MUTATION OPERATIONS
# =============================================================================

@njit(fastmath=True, cache=True)
def adaptive_gaussian_mutation(individual, mutation_rate=0.05, fitness_score=1.0, seed=None):
    """Adaptive Gaussian mutation with fitness-based sigma decay."""
    if seed is not None:
        np.random.seed(seed)
    
    # Adaptive sigma: poor fitness → higher sigma (more exploration)
    sigma = 0.1 + max(0.0, 2.0 - fitness_score) * 0.5
    
    # Mutate random subset of network
    total_params = len(individual)
    num_mutations = max(1, int(total_params * mutation_rate))
    
    for _ in range(num_mutations):
        idx = np.random.randint(0, total_params)
        individual[idx] += np.random.normal(0.0, sigma)
    
    return individual


